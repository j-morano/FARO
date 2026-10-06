import shutil
import argparse
import yaml
import socket
from pathlib import Path
import json
import gzip
import pickle

import utils.data_constants as data_constants



config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser('MultiMAE pre-training script', add_help=False)

parser.add_argument('--batch_size', default=256, type=int,
                    help='Batch size per GPU (default: %(default)s)')
parser.add_argument('--epochs', default=1600, type=int,
                    help='Number of epochs (default: %(default)s)')
parser.add_argument('--save_ckpt_freq', default=20, type=int,
                    help='Checkpoint saving frequency in epochs (default: %(default)s)')

# Task parameters
parser.add_argument('--in_domains', default='rgb-depth-semseg', type=str,
                    help='Input domain names, separated by hyphen (default: %(default)s)')
parser.add_argument('--out_domains', default='rgb-depth-semseg', type=str,
                    help='Output domain names, separated by hyphen (default: %(default)s)')
parser.add_argument('--standardize_depth', action='store_true')
parser.add_argument('--no_standardize_depth', action='store_false', dest='standardize_depth')
parser.set_defaults(standardize_depth=False)
parser.add_argument('--extra_norm_pix_loss', action='store_true')
parser.add_argument('--no_extra_norm_pix_loss', action='store_false', dest='extra_norm_pix_loss')
parser.set_defaults(extra_norm_pix_loss=True)


# Model parameters
parser.add_argument('--model', default='pretrain_multimae_base', type=str, metavar='MODEL',
                    help='Name of model to train (default: %(default)s)')
parser.add_argument('--num_encoded_tokens', default=98, type=int,
                    help='Number of tokens to randomly choose for encoder (default: %(default)s)')
parser.add_argument('--perc_encoded_tokens', default=None, type=float,
                    help='Percentage of tokens to randomly choose for encoder (default: %(default)s)')
parser.add_argument('--num_global_tokens', default=1, type=int,
                    help='Number of global tokens to add to encoder (default: %(default)s)')
parser.add_argument('--patch_size', default=16, type=int,
                    help='Base patch size for image-like modalities (default: %(default)s)')
parser.add_argument('--input_size', default=224, type=int,
                    help='Images input size for backbone (default: %(default)s)')
parser.add_argument('--alphas', type=float, default=1.0,
                    help='Dirichlet alphas concentration parameter (default: %(default)s)')
parser.add_argument('--custom-sampling', action='store_true')
parser.add_argument('--sample-only', type=str, default=None,
                    help='If specified, only sample masks for this task (default: %(default)s)')
parser.add_argument('--sample_tasks_uniformly', default=False, action='store_true',
                    help='Set to True/False to enable/disable uniform sampling over tasks to sample masks for.')

parser.add_argument('--decoder_use_task_queries', default=True, action='store_true',
                    help='Set to True/False to enable/disable adding of task-specific tokens to decoder query tokens')
parser.add_argument('--decoder_use_xattn', default=True, action='store_true',
                    help='Set to True/False to enable/disable decoder cross attention.')
parser.add_argument('--decoder_dim', default=256, type=int,
                    help='Token dimension inside the decoder layers (default: %(default)s)')
parser.add_argument('--decoder_depth', default=2, type=int,
                    help='Number of self-attention layers after the initial cross attention (default: %(default)s)')
parser.add_argument('--decoder_num_heads', default=8, type=int,
                    help='Number of attention heads in decoder (default: %(default)s)')
parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                    help='Drop path rate (default: %(default)s)')

parser.add_argument('--loss_on_unmasked', default=False, action='store_true',
                    help='Set to True/False to enable/disable computing the loss on non-masked tokens')
parser.add_argument('--no_loss_on_unmasked', action='store_false', dest='loss_on_unmasked')
parser.set_defaults(loss_on_unmasked=False)
parser.add_argument('--loss', default='mse', type=str,
                    help='Loss function (default: %(default)s)')

parser.add_argument('--weights', default=None, type=str,
                    help='Path to pretrained model weights (default: %(default)s)')

# Optimizer parameters
parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: %(default)s)')
parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                    help='Optimizer epsilon (default: %(default)s)')
parser.add_argument('--opt_betas', default=[0.9, 0.95], type=float, nargs='+', metavar='BETA',
                    help='Optimizer betas (default: %(default)s)')
parser.add_argument('--clip_grad', type=float, default=None, metavar='CLIPNORM',
                    help='Clip gradient norm (default: %(default)s)')
parser.add_argument('--skip_grad', type=float, default=None, metavar='SKIPNORM',
                    help='Skip update if gradient norm larger than threshold (default: %(default)s)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: %(default)s)')
parser.add_argument('--weight_decay', type=float, default=0.05,
                    help='Weight decay (default: %(default)s)')
parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
    weight decay. We use a cosine schedule for WD.  (Set the same value as args.weight_decay to keep weight decay unchanged)""")
parser.add_argument('--decoder_decay', type=float, default=None, help='decoder weight decay')

parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                    help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256 (default: %(default)s)')
parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                    help='Warmup learning rate (default: %(default)s)')
parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                    help='Lower lr bound for cyclic schedulers that hit 0 (default: %(default)s)')
parser.add_argument('--task_balancer', type=str, default='none',
                    help='Task balancing scheme. One out of [uncertainty, none] (default: %(default)s)')
parser.add_argument('--balancer_lr_scale', type=float, default=1.0,
                    help='Task loss balancer LR scale (if used) (default: %(default)s)')


parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                    help='Epochs to warmup LR, if scheduler supports (default: %(default)s)')
parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                    help='Epochs to warmup LR, if scheduler supports (default: %(default)s)')

parser.add_argument('--fp32_output_adapters', type=str, default='',
                    help='Tasks output adapters to compute in fp32 mode, separated by hyphen.')

# Augmentation parameters
parser.add_argument('--hflip', type=float, default=0.5,
                    help='Probability of horizontal flip (default: %(default)s)')
parser.add_argument('--intensity-shift', type=float, default=0.0,
                    help='Intensity shift (default: %(default)s)')
parser.add_argument('--train_interpolation', type=str, default='bicubic',
                    help='Training interpolation (random, bilinear, bicubic) (default: %(default)s)')
parser.add_argument('--random-crop', type=str, default='0.8--1',
                    help='Random crop (default: %(default)s)')
parser.add_argument('--no-affine', action='store_true')

# Dataset parameters
parser.add_argument('--data_path', default=data_constants.IMAGENET_TRAIN_PATH, type=str, help='dataset path')
parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
parser.add_argument('--fsids', default=None, type=str,
                    help='JSON file with valid file set ids')
parser.add_argument('--filelist', default='./cfgs/all_files.json.gz', type=str,
                    help='File with list of files to use for training')
parser.add_argument('--version', default='v2', type=str,
                    help='Dataset version to use (default: %(default)s)')
# parser.add_argument('--metadata', default='./cfgs/v4_all_metadata.pkl.gz', type=str,
#                     help='File with dataset metadata')
parser.add_argument('--no_fft_loss', action='store_true')


# Misc.
parser.add_argument('--output_dir', default='',
                    help='Path where to save, empty for no saving')
parser.add_argument('--device', default='cuda',
                    help='Device to use for training / testing')

parser.add_argument('--seed', default=0, type=int, help='Random seed ')
parser.add_argument('--resume', default='', help='resume from checkpoint')
parser.add_argument('--auto_resume', action='store_true')
parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
parser.set_defaults(auto_resume=True)

parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
parser.add_argument('--num_workers', default=10, type=int)
parser.add_argument('--pin_mem', action='store_true',
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                    help='')
parser.set_defaults(pin_mem=True)
parser.add_argument('--find_unused_params', action='store_true')
parser.add_argument('--no_find_unused_params', action='store_false', dest='find_unused_params')
parser.set_defaults(find_unused_params=True)

# Wandb logging
parser.add_argument('--log_wandb', default=False, action='store_true',
                    help='Log training and validation metrics to wandb')
parser.add_argument('--no_log_wandb', action='store_false', dest='log_wandb')
parser.set_defaults(log_wandb=False)
parser.add_argument('--wandb_project', default=None, type=str,
                    help='Project name on wandb')
parser.add_argument('--wandb_entity', default=None, type=str,
                    help='User or team name on wandb')
parser.add_argument('--wandb_run_name', default=None, type=str,
                    help='Run name on wandb')
parser.add_argument('--show_user_warnings', default=False, action='store_true')

# Distributed training parameters
parser.add_argument('--world_size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--dist_on_itp', action='store_true')
# parser.add_argument('--no-dist', action='store_true')
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
parser.add_argument('--print-model', action='store_true')
parser.add_argument('--three-d', action='store_true')
parser.add_argument('--center-crop-from', default=None, type=int)
parser.add_argument('--empty-loop', action='store_true')
parser.add_argument('--gpu', nargs='+', type=int, default=[0],
                    help='List of gpu ids to use for training (default: %(default)s)')
parser.add_argument('--dist_port', default=29500, type=int,
                    help='Port number to use for distributed training (default: %(default)s)')

parser.add_argument('--contrastive', action='store_true')
parser.add_argument('--repeat_dataset', default=None, type=int,
                    help='Repeat dataset certain number of times to increase number of iterations per epoch')
parser.add_argument('--patient_balanced', action='store_true')
parser.add_argument('--dino', action='store_true')
parser.add_argument('--overwrite', action='store_true')



# Do we have a config file to parse?
args_config, remaining = config_parser.parse_known_args()
if args_config.config:
    with open(args_config.config, 'r') as f:
        cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)

# The main arg parser parses the rest of the args, the usual
# defaults will have been overridden if config file specified.
args = parser.parse_args(remaining)


if args.perc_encoded_tokens is not None:
    total_tokens = 0
    for domain, size in args.input_size.items():
        current_tokens = 1
        for i, s in enumerate(size):
            current_tokens *= s // args.patch_size[domain][i]
        total_tokens += current_tokens
    print('Computing number of encoded tokens based on percentage:', args.perc_encoded_tokens)
    args.num_encoded_tokens = int(total_tokens * args.perc_encoded_tokens)
    print(f'> Number of encoded tokens: {args.num_encoded_tokens} (out of {total_tokens})')


domains = args.in_domains.split('-')
if isinstance(args.patch_size, int):
    if args.three_d:
        args.patch_size = {d: (args.patch_size, args.patch_size, args.patch_size) for d in domains}
    else:
        args.patch_size = {d: (args.patch_size, args.patch_size) for d in domains}

if isinstance(args.input_size, int):
    if args.three_d:
        args.input_size = {d: (args.input_size, args.input_size, args.input_size) for d in domains}
    else:
        args.input_size = {d: (args.input_size, args.input_size) for d in domains}

args.grid_sizes = {}
for domain, size in args.input_size.items():
    args.grid_sizes[domain] = []
    for i, s in enumerate(size):
        args.grid_sizes[domain].append(s // args.patch_size[domain][i])

fsids = None
if args.fsids is not None:
    with open(args.fsids, 'r') as f:
        fsids = set(json.load(f))

if args_config.config:
    args.wandb_run_name = Path(args_config.config).stem
    # args.output_dir = f'__v2_output/pretrain/{args.wandb_run_name}'
    args.output_dir = f'/raid/optima/__{args.version}_output/pretrain/{args.wandb_run_name}'


if socket.gethostname() == 'hemingway':
    # args.data_path = '/mnt/Data/SSHFS/msc_server/Mini_Datasets/FullVIBES-v2/'
    args.data_path = '/mnt/Data/FullVIBES-v2/v2_new_bscanlayermap/'
    # Add 'mini_' to the filename
    fn = Path(args.filelist)
    fn_name = f'mini_{fn.stem}{fn.suffix}'
    args.filelist = str(fn.parent / fn_name)
    print('> Updated filelist path for hemingway:', args.filelist)
    # if 'per_patient_' in args.filelist:
    #     args.filelist = './cfgs/per_patient_minifiles.json.gz'
    # elif 'all_' in args.filelist:
    #     args.filelist = './cfgs/all_files.json.gz'
    # elif 'databank' in args.filelist:
    #     args.filelist = './cfgs/mini_training_databank.pkl'
    #     args.data_path = '/mnt/Data/FullVIBES-v2/v2_new_bscanlayermap/'
    # else:
    #     raise ValueError('Unknown filelist specified in config')
    # args.weights = '_weights/MIRAGE-Base.pth'
    # if args.contrastive:
    #     if 'width' in args.model:
    #         args.model = 'pretrain_miragev2conwidth_base'
    #     else:
    #         args.model = 'pretrain_miragev2con_base'
    # else:
    #     if 'wt' in args.model:
    #         args.model = 'pretrain_miragev2wt_base'
    #     else:
    #         args.model = 'pretrain_miragev2_base'
    args.batch_size = 2
    args.num_workers = 2
    args.output_dir = f'__{args.version}_output/pretrain/{args.wandb_run_name}'
    args.repeat_dataset = 10

# Print configuration
args_dict = vars(args)
# Order alphabetically
args_dict = dict(sorted(args_dict.items()))
print(json.dumps(args_dict, indent=2))

# Create output directory
if args.output_dir:
    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        print(f'> Warning: removing contents of {args.output_dir}')
        for f in output_dir.glob('*'):
            try:
                if f.is_file():
                    f.unlink()
                elif f.is_dir():
                    shutil.rmtree(f)
            except FileNotFoundError:
                pass
    (output_dir / 'debug').mkdir(parents=True, exist_ok=True)

# Save configuration
with open(Path(args.output_dir, 'config.json'), 'w') as f:
    json.dump(vars(args), f, indent=2)

# Add fsids to args
args.fsids = fsids

# if not args.patient_balanced:
#     with gzip.open(args.metadata, 'rb') as f:
#         print('> Loading metadata from', args.metadata)
#         args.metadata = pickle.load(f)
#         print('> Number of entries in metadata:', len(args.metadata))
