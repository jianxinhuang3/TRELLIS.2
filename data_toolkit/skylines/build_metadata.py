"""
T0 step 1: build/refresh metadata.csv for the skylines_50k extraction root.

Reads skylines_50k_labels.tsv, filters the requested instances (sha256 naming
convention: basename of glb without extension, e.g. 'xxxx_obj_0'), and writes
one row per instance with the label geometry fields.

Usage:
    python build_metadata.py --instances ddb33c1bdba1999ac314153ea8990cbd_obj_0
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import argparse
import pandas as pd

from data_toolkit.skylines import common


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels_tsv', type=str, default=common.LABELS_TSV,
                        help='Path to skylines_50k_labels.tsv')
    parser.add_argument('--instances', type=str, required=True,
                        help='Comma separated sha256 list, or a file with one sha256 per line')
    opt = parser.parse_args()

    instances = common.parse_instances(opt.instances)
    labels = pd.read_csv(opt.labels_tsv, sep='\t')
    labels['sha256'] = labels['glb_path'].map(lambda p: os.path.splitext(os.path.basename(p))[0])

    os.makedirs(common.EXTRACT_ROOT, exist_ok=True)

    selected = labels[labels['sha256'].isin(instances)]
    missing = set(instances) - set(selected['sha256'])
    if missing:
        raise ValueError(f'instances not found in labels tsv: {sorted(missing)}')

    for _, row in selected.iterrows():
        if not os.path.exists(row['glb_path']):
            raise FileNotFoundError(f"glb not found: {row['glb_path']}")
        common.update_metadata(
            row['sha256'],
            glb_path=row['glb_path'],
            height=float(row['height']),
            size_x=float(row['size_x']),
            size_z=float(row['size_z']),
            aesthetic_score=5.0,
        )
        print(f"[build_metadata] {row['sha256']}: glb={row['glb_path']} "
              f"size_x={row['size_x']} size_z={row['size_z']} height={row['height']}")

    print(f'[build_metadata] wrote {common.metadata_path()} ({len(selected)} instances)')
