"""
Rebuild the global metadata.csv after (or during) a batch extraction run.

The batch scheduler (extract_all.py) deliberately does NOT touch the shared
metadata.csv (per-sample read-modify-write of a 44k-row CSV would race across
shards). This script rebuilds it in one vectorized pass:

    - base fields (glb_path, height, size_x, size_z, aesthetic_score) from
      labels.tsv
    - stage flags (ss/shape/tex_latent_encoded, cond_rendered, proxy_built)
      from artifact existence (file present + non-empty == done). Flags are
      True or NaN -- never False -- so that the training-side filters
      (`== True` and `.notna()`) both behave correctly
    - token counts from the per-shard records JSONL when available, else the
      previous metadata.csv, else by reading the shape latent npz (pooled)
    - pbr_latent_encoded / pbr_latent_tokens mirror the tex latent (the
      svpbr dataset filters on these keys)
    - metadata.csv symlinks in proxy/, render_cond/, ss_latents/<name>/,
      shape_latents/<name>/, tex_latents/<name>/ are (re)created
    - per-shard failures_rank*.csv are merged into failures.csv

Usage:
    python aggregate_metadata.py [--labels_tsv ...] [--num_workers 16]
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import json
import glob
import argparse
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

from data_toolkit.skylines import common
from data_toolkit.skylines.extract_all import scan_status, LOG_ROOT


def _read_tokens(sha256):
    try:
        with np.load(common.shape_latent_path(sha256)) as f:
            return sha256, int(f['coords'].shape[0])
    except Exception:
        return sha256, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels_tsv', type=str, default=common.LABELS_TSV)
    parser.add_argument('--num_workers', type=int, default=16)
    opt = parser.parse_args()

    labels = pd.read_csv(opt.labels_tsv, sep='\t')
    labels['sha256'] = labels['glb_path'].map(
        lambda p: os.path.splitext(os.path.basename(p))[0])
    labels = labels.drop_duplicates('sha256').set_index('sha256').sort_index()
    print(f'[aggregate] labels: {len(labels)} instances')

    status = scan_status(list(labels.index))
    n_enc = sum(s['encode'] for s in status.values())
    n_rnd = sum(s['render'] for s in status.values())
    n_prx = sum(s['proxy'] for s in status.values())
    print(f'[aggregate] artifacts: encode={n_enc} render={n_rnd} proxy={n_prx}')

    # ---- token counts: records JSONL > previous metadata.csv > npz read ----
    tokens = {}
    for path in sorted(glob.glob(os.path.join(LOG_ROOT, 'records_rank*.jsonl'))):
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                if 'shape_latent_tokens' in rec:
                    tokens[rec['sha256']] = int(rec['shape_latent_tokens'])
    if os.path.exists(common.metadata_path()):
        prev = pd.read_csv(common.metadata_path()).set_index('sha256')
        if 'shape_latent_tokens' in prev.columns:
            for sha256, v in prev['shape_latent_tokens'].dropna().items():
                tokens.setdefault(sha256, int(v))
    need_read = [s for s in labels.index if status[s]['encode'] and s not in tokens]
    if need_read:
        print(f'[aggregate] reading token counts from {len(need_read)} npz files ...')
        with ProcessPoolExecutor(max_workers=opt.num_workers) as ex:
            for sha256, n in ex.map(_read_tokens, need_read, chunksize=64):
                if n is not None:
                    tokens[sha256] = n

    # ---- assemble (True/NaN flags; never False, see module docstring) ----
    df = pd.DataFrame({'sha256': list(labels.index)}).set_index('sha256')
    df['glb_path'] = labels['glb_path']
    df['height'] = labels['height'].astype(float)
    df['size_x'] = labels['size_x'].astype(float)
    df['size_z'] = labels['size_z'].astype(float)
    df['aesthetic_score'] = 5.0
    enc = pd.Series({s: True if status[s]['encode'] else np.nan for s in df.index})
    tok = pd.Series({s: tokens.get(s, np.nan) for s in df.index})
    df['ss_latent_encoded'] = enc
    df['shape_latent_encoded'] = enc
    df['shape_latent_tokens'] = tok
    df['tex_latent_encoded'] = enc
    df['tex_latent_tokens'] = tok
    df['pbr_latent_encoded'] = enc
    df['pbr_latent_tokens'] = tok
    df['cond_rendered'] = pd.Series(
        {s: True if status[s]['render'] else np.nan for s in df.index})
    df['proxy_built'] = pd.Series(
        {s: True if status[s]['proxy'] else np.nan for s in df.index})

    n_full = int((df['shape_latent_encoded'].notna() & df['cond_rendered'].notna()
                  & df['proxy_built'].notna()).sum())
    missing_tok = int((enc.notna() & tok.isna()).sum())
    if missing_tok:
        print(f'[aggregate] WARNING: {missing_tok} encoded instances without '
              f'readable token count (left NaN, will be filtered by training)')

    os.makedirs(common.EXTRACT_ROOT, exist_ok=True)
    df.reset_index().to_csv(common.metadata_path(), index=False)
    print(f'[aggregate] wrote {common.metadata_path()}: {len(df)} rows, '
          f'{n_full} fully complete')

    # ---- symlinks expected by the training datasets ----
    for sub in ['proxy', 'render_cond',
                os.path.join('ss_latents', common.SS_LATENT_NAME),
                os.path.join('shape_latents', common.SHAPE_LATENT_NAME),
                os.path.join('tex_latents', common.TEX_LATENT_NAME)]:
        d = os.path.join(common.EXTRACT_ROOT, sub)
        os.makedirs(d, exist_ok=True)
        link = os.path.join(d, 'metadata.csv')
        if not os.path.islink(link) and not os.path.exists(link):
            os.symlink(common.metadata_path(), link)
            print(f'[aggregate] symlink created: {link}')

    # ---- merge failure logs (drop entries that later succeeded on retry) ----
    parts = sorted(glob.glob(os.path.join(LOG_ROOT, 'failures_rank*.csv')))
    if parts:
        merged = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        merged = merged.drop_duplicates()
        stage_key = {'encode': 'encode', 'render': 'render', 'proxy': 'proxy'}
        still_failed = merged.apply(
            lambda r: r['sha256'] not in status
            or not status[r['sha256']].get(stage_key.get(r['stage'], ''), False),
            axis=1)
        merged = merged[still_failed]
        merged.to_csv(os.path.join(LOG_ROOT, 'failures.csv'), index=False)
        print(f'[aggregate] failures.csv: {len(merged)} rows '
              f'({merged["sha256"].nunique()} instances) from {len(parts)} shards')
    print('[aggregate] done')


if __name__ == '__main__':
    main()
