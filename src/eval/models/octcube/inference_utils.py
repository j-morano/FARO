import os
import torch
import argparse
import numpy as np

import models.octcube.models_vit_st_flash_attn as models_vit_st_flash_attn
from models.octcube.util.misc import interpolate_pos_embed, interpolate_temporal_pos_embed

disease_abbreviation = {
    0: 'Normal',
    1: 'DME',
    2: 'AMD',
    3: 'POAG',
    4: 'EPM',
    5: 'DR',
    6: 'VD',
    7: 'RAO/RVO',
    8: 'RNV',
}

def process_dicom_array(dicom_array, val_transform):
    dicom_tensor = torch.tensor(dicom_array).unsqueeze(0)
    dicom_tensor = val_transform({"pixel_values": dicom_tensor})["pixel_values"]
    return dicom_tensor, dicom_tensor.shape


def load_model(args, model_without_ddp):
    if args.ckpt:
        checkpoint = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        new_num_spatial_tokens = args.input_size // args.patch_size
        interpolate_pos_embed(model_without_ddp, checkpoint['model'], new_num_spatial_tokens)
        interpolate_temporal_pos_embed(model_without_ddp, checkpoint['model'])
        msg = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        print("Load checkpoint %s" % args.ckpt)
        print(msg)
    else:
        print("No checkpoint for loading")


def create_models(args):
    if args.model_type == '3D_st_flash_attn':
        print('Use 3D spatio-temporal model w/ flash attention')
        model = models_vit_st_flash_attn.__dict__[args.model](
                num_frames=args.num_frames,
                t_patch_size=args.t_patch_size,
                img_size=args.input_size,
                num_classes=args.nb_classes,
                drop_path_rate=args.drop_path,
                global_pool=args.global_pool,
                sep_pos_embed=args.sep_pos_embed,
                cls_embed=args.cls_embed,
                use_flash_attention=True
            )

    model = model.cuda()
    load_model(args, model)
    return model


def parse_all_output(pred_output_cache):
    highest_disease_classes = np.argmax(pred_output_cache[:, 1])
    highest_disease_prob = pred_output_cache[highest_disease_classes, 1]
    if highest_disease_prob > 0.5:
        disease_flag = True
    else:
        disease_flag = False
    all_output = 'Disease probability: (Disease Name: Probability) \n'
    for i in range(len(disease_abbreviation)):
        if i == 0:
            if disease_flag:
                all_output += f"{disease_abbreviation[i]}: {1 - highest_disease_prob:.3f}        "
            else:
                all_output += f"{disease_abbreviation[i]}: {np.mean(pred_output_cache[:, 0]):.3f}         "
        else:
            all_output += f"{disease_abbreviation[i]}: {pred_output_cache[i-1, 1]:.3f}       "
        if (i + 1) % 3 == 0:
            all_output += ''
    return all_output
