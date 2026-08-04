"""
Full-scale batch scheduler for the SatSkylines T0 extraction pipeline.

Scales the verified single-sample flow (voxelize_and_encode -> render_roof ->
build_proxy) to all ~44k instances of skylines_50k_labels.tsv. The single
sample scripts are imported as libraries and NOT modified; encoder weights are
loaded once per shard process.

Features:
    - resume: scans artifact directories at startup and skips finished stages
      (file existence + non-zero size is the source of truth)
    - sharding: --rank/--world_size splits the instance list; each shard is an
      independent process bound to one GPU via CUDA_VISIBLE_DEVICES
    - failure isolation: per-instance/per-stage errors are appended to
      outputs/extract_logs/failures_rank{r}.csv and the run continues;
      CUDA OOM triggers an empty_cache + single retry
    - watchdog: SIGALRM-based per-stage timeout (default 600 s). NOTE: the
      alarm can only interrupt Python bytecode, a hang inside native code
      needs an external kill (resume makes that cheap)
    - metadata: common.update_metadata is redirected to an append-only
      per-shard JSONL (records_rank{r}.jsonl); the shared metadata.csv is NOT
      touched during the run -> no cross-shard races. Run aggregate_metadata.py
      afterwards to rebuild metadata.csv.
    - progress: tqdm + progress_rank{r}.json refreshed after every instance

Usage (usually via scripts/skylines/extract_all.sh):
    python extract_all.py --rank 0 --world_size 4 [--limit N]
        [--instances file_or_csv] [--renderer auto] [--stage_timeout 600]
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import io
import json
import time
import signal
import argparse
import traceback
import contextlib
import fcntl
import subprocess
import pandas as pd

from data_toolkit.skylines import common

LOG_ROOT = os.path.join(common.TRELLIS2_ROOT, 'outputs', 'extract_logs')
BLENDER_DIR = '/tmp/blender-4.5.1-linux-x64'
BLENDER_BIN = os.path.join(BLENDER_DIR, 'blender')
BLENDER_TARBALLS = [
    '/data5/jianxin/tools/blender-4.5.1-linux-x64.tar.xz',
    '/tmp/blender-4.5.1-linux-x64.tar.xz',
]
STAGES = ('encode', 'render', 'proxy')


# --------------------------- status scan (resume) ---------------------------

def _nonempty_files(directory):
    """Set of file names with size > 0 in a directory (empty set if missing)."""
    if not os.path.isdir(directory):
        return set()
    out = set()
    with os.scandir(directory) as it:
        for e in it:
            try:
                if e.is_file(follow_symlinks=False) and e.stat().st_size > 0:
                    out.add(e.name)
            except OSError:
                pass
    return out


def scan_status(instances):
    """
    For each sha256 return {'encode': bool, 'render': bool, 'proxy': bool}
    from artifact existence (file present and non-empty == stage done).
    """
    ss = _nonempty_files(os.path.join(common.EXTRACT_ROOT, 'ss_latents', common.SS_LATENT_NAME))
    shape = _nonempty_files(os.path.join(common.EXTRACT_ROOT, 'shape_latents', common.SHAPE_LATENT_NAME))
    tex = _nonempty_files(os.path.join(common.EXTRACT_ROOT, 'tex_latents', common.TEX_LATENT_NAME))
    aux = _nonempty_files(os.path.join(common.EXTRACT_ROOT, 'aux'))
    proxy = _nonempty_files(os.path.join(common.EXTRACT_ROOT, 'proxy'))
    render_root = os.path.join(common.EXTRACT_ROOT, 'render_cond')

    render_frames = {f'{i:03d}.png' for i in range(8)} | {'transforms.json'}
    status = {}
    for sha256 in instances:
        npz = f'{sha256}.npz'
        enc = npz in ss and npz in shape and npz in tex and npz in aux
        rnd = render_frames <= _nonempty_files(os.path.join(render_root, sha256))
        status[sha256] = {'encode': enc, 'render': rnd, 'proxy': npz in proxy}
    return status


# --------------------------- blender availability ---------------------------

def ensure_blender():
    """
    /tmp is wiped on reboot; re-extract blender from a tarball if missing.
    flock guards against concurrent extraction by parallel shards.
    Returns True if the blender binary is available.
    """
    if os.path.exists(BLENDER_BIN):
        return True
    tarball = next((t for t in BLENDER_TARBALLS if os.path.exists(t)), None)
    if tarball is None:
        print('[extract_all] WARNING: blender missing and no tarball found; '
              'renders will fall back to nvdiffrast')
        return False
    lock_path = '/tmp/.blender_install.lock'
    with open(lock_path, 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)          # one shard extracts, others wait
        if not os.path.exists(BLENDER_BIN):
            print(f'[extract_all] extracting {tarball} -> /tmp ...')
            subprocess.run(['tar', '-xJf', tarball, '-C', '/tmp'], check=True)
        fcntl.flock(lock, fcntl.LOCK_UN)
    return os.path.exists(BLENDER_BIN)


# --------------------------- per-stage watchdog ---------------------------

class StageTimeout(Exception):
    pass


@contextlib.contextmanager
def stage_timeout(seconds):
    def _handler(signum, frame):
        raise StageTimeout(f'stage exceeded {seconds}s')
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# --------------------------- shard-local metadata records ---------------------------

def make_recorder(rank):
    """Replace common.update_metadata with an append-only per-shard JSONL."""
    path = os.path.join(LOG_ROOT, f'records_rank{rank}.jsonl')

    def record(sha256, **fields):
        with open(path, 'a') as f:
            f.write(json.dumps({'sha256': sha256, **fields}) + '\n')
    return record


# --------------------------- driver ---------------------------

def run_stage(fn, timeout_s, *args, **kwargs):
    """Run one stage with watchdog + one OOM retry. Raises on failure."""
    import torch
    try:
        with stage_timeout(timeout_s):
            return fn(*args, **kwargs)
    except torch.cuda.OutOfMemoryError:
        print('[extract_all] CUDA OOM, empty_cache + retry once')
        torch.cuda.empty_cache()
        with stage_timeout(timeout_s):
            return fn(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--labels_tsv', type=str, default=common.LABELS_TSV)
    parser.add_argument('--instances', type=str, default=None,
                        help='Optional subset: comma separated sha256 list or file '
                             '(default: all rows of labels_tsv)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Process at most N instances of this shard (smoke tests)')
    parser.add_argument('--renderer', type=str, default='auto',
                        choices=['auto', 'blender', 'nvdiffrast'])
    parser.add_argument('--stage_timeout', type=int, default=600,
                        help='Per-instance per-stage timeout in seconds')
    parser.add_argument('--seed', type=int, default=0)
    opt = parser.parse_args()
    assert 0 <= opt.rank < opt.world_size

    os.makedirs(LOG_ROOT, exist_ok=True)
    tag = f'rank{opt.rank}'

    # ---- instance list: labels.tsv is the source of truth ----
    labels = pd.read_csv(opt.labels_tsv, sep='\t')
    labels['sha256'] = labels['glb_path'].map(
        lambda p: os.path.splitext(os.path.basename(p))[0])
    labels = labels.drop_duplicates('sha256').set_index('sha256').sort_index()
    if opt.instances:
        subset = common.parse_instances(opt.instances)
        missing_lbl = [s for s in subset if s not in labels.index]
        assert not missing_lbl, f'not in labels tsv: {missing_lbl[:5]}'
        labels = labels.loc[subset].sort_index()

    all_shas = list(labels.index)
    glb_missing = [s for s in all_shas if not os.path.exists(labels.loc[s, 'glb_path'])]
    if glb_missing and opt.rank == 0:
        pd.DataFrame({'sha256': glb_missing}).to_csv(
            os.path.join(LOG_ROOT, 'glb_missing.csv'), index=False)
    avail = [s for s in all_shas if s not in set(glb_missing)]

    # ---- resume: drop instances whose three stages are all complete ----
    t0 = time.time()
    status = scan_status(avail)
    pending = [s for s in avail if not all(status[s].values())]
    print(f'[{tag}] labels={len(all_shas)} glb_missing={len(glb_missing)} '
          f'complete={len(avail) - len(pending)} pending={len(pending)} '
          f'(scan {time.time() - t0:.1f}s)')

    # ---- shard split (interleaved on the sorted list -> balanced) ----
    shard = pending[opt.rank::opt.world_size]
    if opt.limit is not None:
        shard = shard[:opt.limit]
    print(f'[{tag}] shard size = {len(shard)} '
          f'(world_size={opt.world_size}, limit={opt.limit})')
    if not shard:
        print(f'[{tag}] nothing to do')
        return

    # ---- redirect per-sample metadata updates to the shard JSONL ----
    common.update_metadata = make_recorder(opt.rank)

    # o_voxel's mipmap builder rejects non-square / non-power-of-two textures
    # (~30% of skylines assets); sanitize them right after GLB loading.
    _load_raw = common.load_normalized_geoms

    def _load_sanitized(glb_path):
        geoms, norm = _load_raw(glb_path)
        return common.sanitize_pbr_textures(geoms), norm
    common.load_normalized_geoms = _load_sanitized

    blender_ok = ensure_blender()
    renderer = opt.renderer
    if renderer == 'auto' and not blender_ok:
        renderer = 'nvdiffrast'

    # heavy imports after arg parsing so --help stays fast
    import torch
    from tqdm import tqdm
    import trellis2.models as models
    from data_toolkit.skylines import voxelize_and_encode, render_roof, build_proxy

    need_encode = any(not status[s]['encode'] for s in shard)
    shape_enc = ss_enc = tex_enc = None
    if need_encode:                                # load weights once per shard
        shape_enc = models.from_pretrained(common.SHAPE_ENC_PRETRAINED).eval().cuda()
        ss_enc = models.from_pretrained(common.SS_ENC_PRETRAINED).eval().cuda()
        tex_enc = models.from_pretrained(common.TEX_ENC_PRETRAINED).eval().cuda()
        print(f'[{tag}] encoders loaded on {torch.cuda.get_device_name(0)}')

    failures_path = os.path.join(LOG_ROOT, f'failures_{tag}.csv')
    progress_path = os.path.join(LOG_ROOT, f'progress_{tag}.json')
    n_done, n_failed = 0, 0
    stage_time = {s: [0.0, 0] for s in STAGES}     # total seconds, count
    t_start = time.time()

    def fail(sha256, stage, exc):
        nonlocal n_failed
        n_failed += 1
        msg = ''.join(traceback.format_exception_only(type(exc), exc)).strip()
        new = not os.path.exists(failures_path)
        with open(failures_path, 'a') as f:
            if new:
                f.write('sha256,stage,error\n')
            f.write(f'{sha256},{stage},"{msg[:500].replace(chr(34), chr(39))}"\n')
        print(f'[{tag}] FAIL {sha256} @{stage}: {msg[:200]}')

    pbar = tqdm(shard, desc=tag, dynamic_ncols=True)
    for i, sha256 in enumerate(pbar):
        glb_path = labels.loc[sha256, 'glb_path']
        st = status[sha256]
        ok = True
        # keep worker prints out of the way; tqdm goes to stderr
        for stage, done in (('encode', st['encode']),
                            ('render', st['render']),
                            ('proxy', st['proxy'])):
            if done or not ok:
                continue
            t = time.time()
            try:
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    if stage == 'encode':
                        run_stage(voxelize_and_encode.process_one, opt.stage_timeout,
                                  sha256, glb_path, shape_enc, ss_enc, tex_enc)
                    elif stage == 'render':
                        run_stage(render_roof.render_one, opt.stage_timeout,
                                  sha256, glb_path, renderer, BLENDER_BIN, opt.seed)
                    elif stage == 'proxy':
                        run_stage(build_proxy.build_one, opt.stage_timeout,
                                  sha256, opt.seed)
                stage_time[stage][0] += time.time() - t
                stage_time[stage][1] += 1
            except Exception as e:
                fail(sha256, stage, e)
                ok = False
                torch.cuda.empty_cache()
        if ok:
            n_done += 1

        elapsed = time.time() - t_start
        remain = len(shard) - (i + 1)
        eta = elapsed / (i + 1) * remain
        pbar.set_postfix(done=n_done, failed=n_failed)
        with open(progress_path, 'w') as f:
            json.dump({
                'rank': opt.rank, 'world_size': opt.world_size,
                'shard_total': len(shard), 'processed': i + 1,
                'done': n_done, 'failed': n_failed, 'remaining': remain,
                'elapsed_sec': round(elapsed, 1), 'eta_sec': round(eta, 1),
                'sec_per_instance': round(elapsed / (i + 1), 2),
                'stage_avg_sec': {s: round(v[0] / v[1], 2) if v[1] else None
                                  for s, v in stage_time.items()},
                'updated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            }, f, indent=2)

    print(f'[{tag}] finished: done={n_done} failed={n_failed} '
          f'({(time.time() - t_start) / 60:.1f} min)')
    for s, (tot, cnt) in stage_time.items():
        if cnt:
            print(f'[{tag}]   {s}: {tot / cnt:.2f} s/instance over {cnt} runs')


if __name__ == '__main__':
    main()
