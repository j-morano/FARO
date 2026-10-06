"""
Computational efficiency comparison: inference time and memory usage
as a function of the number of B-scans per volume, for 3D-capable
foundation models (e.g. CLOSER vs. OCTCube).

Each model uses its own default input_size (as resolved by its
fm_config entry) unless explicitly overridden via --input_size.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import pandas as pd

from fm_config import fm_config_factory


def get_args():
    parser = argparse.ArgumentParser(description='Computational efficiency benchmark')
    parser.add_argument(
        '--weights', type=str, required=True,
        help='Model key registered in fm_config_factory, e.g. "closer" or "octcube".',
    )
    parser.add_argument(
        '--b_scan_counts', type=str, default='7,19,25,49,97,128,256,512',
        help='Comma-separated list of B-scan counts to benchmark.',
    )
    parser.add_argument(
        '--input_size', type=int, default=None,
        help='Input image size. If None, uses the model config\'s own '
             'default (e.g. 512 for CLOSER, 256 for OCTCube).',
    )
    parser.add_argument('--n_warmup', type=int, default=3,
                        help='Number of warmup iterations before timing.')
    parser.add_argument('--n_repeats', type=int, default=10,
                        help='Number of timed repeats, averaged.')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--pool', type=str, default=None)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--output_dir', type=str, default='./__output/efficiency')
    parser.add_argument('--fill', type=int, default=0)
    return parser.parse_args()


def build_model(args):
    """
    Instantiates the model via fm_config_factory, letting each model's
    own config resolve its default input_size if the user did not
    explicitly pass one.
    """
    model_config = None
    model_name = None
    for kw in fm_config_factory.keys():
        if kw.lower() == args.weights.lower():
            model_config = fm_config_factory[kw](args)
            model_name = kw
            break
    if model_config is None:
        raise ValueError(f"Unknown model: {args.weights}")

    model = model_config.model
    model.to(args.device)
    model.eval()

    # model_config.args holds the resolved input_size (the model's own
    # default if the user passed None), so read it back from there
    resolved_input_size = model_config.args.input_size

    return model, model_name, resolved_input_size


def benchmark_single_config(model, n_bscans, input_size, device, n_warmup, n_repeats):
    """
    Runs inference on a dummy volume of shape [1, 1, n_bscans, H, W],
    measuring wall-clock time and peak GPU memory. Uses autocast to
    match the model's expected mixed-precision usage pattern.
    """
    dummy_volume = torch.randn(1, 1, n_bscans, input_size, input_size, device=device)

    with torch.no_grad():
        for _ in range(n_warmup):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                _ = model(dummy_volume)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    times = []
    with torch.no_grad():
        for _ in range(n_repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                _ = model(dummy_volume)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    mean_time = float(np.mean(times))
    std_time = float(np.std(times))

    return {
        'n_bscans': n_bscans,
        'input_size': input_size,
        'mean_time_s': mean_time,
        'std_time_s': std_time,
        'peak_mem_gb': peak_mem_gb,
    }


def main():
    args = get_args()
    device = torch.device(args.device)
    b_scan_counts = [int(x) for x in args.b_scan_counts.split(',')]

    model, model_name, resolved_input_size = build_model(args)
    print(f'Model: {model_name} | Using input_size={resolved_input_size} (model default)')

    n_parameters = sum(p.numel() for p in model.parameters())
    print(f'Params: {n_parameters:,}')

    results = []
    for n_bscans in b_scan_counts:
        print(f'Benchmarking {model_name} with {n_bscans} B-scans...')
        try:
            result = benchmark_single_config(
                model, n_bscans, resolved_input_size, device,
                args.n_warmup, args.n_repeats,
            )
            result['model'] = model_name
            results.append(result)
            print(f'  Mean time: {result["mean_time_s"]:.4f}s ± {result["std_time_s"]:.4f}s | '
                  f'Peak mem: {result["peak_mem_gb"]:.2f} GB')
        except torch.cuda.OutOfMemoryError:
            print(f'  OOM at {n_bscans} B-scans — recording and stopping further increases.')
            torch.cuda.empty_cache()
            results.append({
                'model': model_name, 'n_bscans': n_bscans, 'input_size': resolved_input_size,
                'mean_time_s': None, 'std_time_s': None, 'peak_mem_gb': None,
            })
            break

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    out_fn = output_dir / f'{model_name}_efficiency.csv'
    df.to_csv(out_fn, index=False)
    print(f'Saved results to {out_fn}')


if __name__ == '__main__':
    main()
