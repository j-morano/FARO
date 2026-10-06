#!/usr/bin/env python3
"""
Recursively rename all files matching 'results*.csv' to '_old_results*.csv'
under a given root directory.

Usage:
    python rename_results.py /mnt/Data/SSHFS/msc_server/MIRAGE_future/eval/__output/seg/SOTA_comparison_new/0
    python rename_results.py <root_dir> --dry-run
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=str, help='Root directory to search recursively.')
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print what would be renamed without actually renaming anything.'
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise ValueError(f'Path "{root}" does not exist.')

    matches = sorted(root.rglob('results*.csv'))
    # Avoid double-prefixing files that are already renamed (e.g., from a previous run).
    matches = [f for f in matches if not f.name.startswith('_old_')]

    if not matches:
        print('No matching files found.')
        return

    print(f'Found {len(matches)} file(s) to rename:\n')
    for fn in matches:
        new_fn = fn.with_name(f'_old_{fn.name}')
        print(f'  {fn}  ->  {new_fn.name}')
        if not args.dry_run:
            fn.rename(new_fn)

    if args.dry_run:
        print('\nDry run only — no files were renamed. Re-run without --dry-run to apply.')
    else:
        print('\nDone.')


if __name__ == '__main__':
    main()
