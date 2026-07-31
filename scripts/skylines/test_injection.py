"""
Unit tests for the decoupled cross-attention injection layers (task T1/T2 prep).

Tests:
    1. gate=0 equivalence: injected model output == bare base model output.
    2. Parameter stats: trainable params < 80M per model, base fully frozen.
    3. Gradient flow: gate / QKV / GeoTokenizer receive gradients.
    4. CFG drop path: forward with proxy_latent=None runs without error.

Run:
    source /data5/jianxin/anaconda3/bin/activate trellis2
    cd /data5/jianxin/TRELLIS.2
    export PYTHONPATH=/data5/jianxin/TRELLIS.2:$PYTHONPATH
    python scripts/skylines/test_injection.py
"""
import os
import sys
import gc
import traceback

import torch

sys.path.insert(0, '/data5/jianxin/TRELLIS.2')

from trellis2 import models
from trellis2.modules.sparse import SparseTensor

WEIGHTS = '/data5/jianxin/TRELLIS.2/weights/TRELLIS.2-4B/ckpts'
SS_FLOW = f'{WEIGHTS}/ss_flow_img_dit_1_3B_64_bf16'
SHAPE_FLOW = f'{WEIGHTS}/slat_flow_img2shape_dit_1_3B_512_bf16'

PARAM_BUDGET = 80_000_000

results = {}


def free():
    gc.collect()
    torch.cuda.empty_cache()


def param_stats(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    base_all_frozen = all(not p.requires_grad for p in model.base.parameters())
    return trainable, frozen, base_all_frozen


def make_ss_inputs(B=2, device='cuda'):
    torch.manual_seed(0)
    x = torch.randn(B, 8, 16, 16, 16, device=device)
    t = torch.full((B,), 500.0, device=device)
    cond = torch.randn(B, 257, 1024, device=device)
    proxy_latent = torch.randn(B, 8, 16, 16, 16, device=device)
    return x, t, cond, proxy_latent


def make_slat_inputs(B=2, tokens=(700, 900), in_channels=32, device='cuda'):
    torch.manual_seed(0)
    coords_list = []
    feats_list = []
    for i, n in enumerate(tokens[:B]):
        # unique random coords in [0, 64)^3
        flat = torch.randperm(64 ** 3)[:n]
        c = torch.stack([flat // (64 * 64), (flat // 64) % 64, flat % 64], dim=-1).int()
        coords_list.append(torch.cat([torch.full((n, 1), i, dtype=torch.int32), c], dim=-1))
        feats_list.append(torch.randn(n, in_channels))
    coords = torch.cat(coords_list).to(device)
    feats = torch.cat(feats_list).to(device)
    x = SparseTensor(feats=feats, coords=coords)
    B_eff = len(tokens[:B])
    t = torch.full((B_eff,), 500.0, device=device)
    cond = torch.randn(B_eff, 257, 1024, device=device)
    proxy_latent = torch.randn(B_eff, 8, 16, 16, 16, device=device)
    return x, t, cond, proxy_latent


def test_ss():
    print('\n' + '=' * 80)
    print('SS flow: InjectedSparseStructureFlowModel')
    print('=' * 80)
    x, t, cond, proxy_latent = make_ss_inputs()

    # --- Test 1: gate=0 equivalence against bare base model ---
    bare = models.from_pretrained(SS_FLOW).cuda().eval()
    with torch.no_grad():
        y_base = bare(x, t, cond)
    del bare
    free()

    injected = models.InjectedSparseStructureFlowModel(pretrained_base=SS_FLOW).cuda()
    injected.base.eval()
    with torch.no_grad():
        y_inj = injected(x, t, cond, proxy_latent=proxy_latent)
    max_diff = (y_inj - y_base).abs().max().item()
    print(f'[Test 1] gate=0 equivalence: max|delta| = {max_diff:.3e}')
    assert max_diff < 1e-3, f'Equivalence failed: {max_diff}'
    results['ss/test1_equivalence'] = f'PASS (max|delta|={max_diff:.3e})'

    # --- Test 2: parameter stats ---
    trainable, frozen, base_all_frozen = param_stats(injected)
    print(f'[Test 2] trainable params: {trainable:,} ({trainable/1e6:.2f}M), '
          f'frozen params: {frozen:,} ({frozen/1e6:.2f}M), base frozen: {base_all_frozen}')
    assert trainable < PARAM_BUDGET, f'Trainable params exceed budget: {trainable}'
    assert base_all_frozen, 'Base model has trainable params'
    results['ss/test2_params'] = f'PASS (trainable={trainable/1e6:.2f}M < 80M, base frozen)'

    # --- Test 3: gradient flow ---
    x_g = x.clone()
    pred = injected(x_g, t, cond, proxy_latent=proxy_latent)
    loss = pred.float().sum()
    loss.backward()
    n_gate_grad, n_qkv_grad = 0, 0
    for i, inj in enumerate(injected.injectors):
        assert inj.gate.grad is not None, f'injector {i} gate grad is None'
        assert inj.to_q.weight.grad is not None, f'injector {i} to_q grad is None'
        assert inj.to_kv.weight.grad is not None, f'injector {i} to_kv grad is None'
        assert inj.to_out.weight.grad is not None, f'injector {i} to_out grad is None'
        if inj.gate.grad.abs().item() > 0:
            n_gate_grad += 1
        if inj.to_q.weight.grad.abs().max().item() > 0:
            n_qkv_grad += 1
    geo_grads = [p.grad is not None for p in injected.geo_tokenizer.parameters()]
    print(f'[Test 3] gate grads non-None: 30/30, nonzero: {n_gate_grad}/30; '
          f'QKV grads non-None: 30/30 (nonzero expected only when gate!=0: {n_qkv_grad}/30); '
          f'GeoTokenizer grads non-None: {sum(geo_grads)}/{len(geo_grads)}')
    assert n_gate_grad == len(injected.injectors), 'some gate grads are zero'
    assert all(geo_grads), 'GeoTokenizer params missing grads'
    # with a non-zero gate the QKV path must receive non-zero grads
    injected.zero_grad(set_to_none=True)
    with torch.no_grad():
        for inj in injected.injectors:
            inj.gate.fill_(0.5)
    pred = injected(x, t, cond, proxy_latent=proxy_latent)
    pred.float().sum().backward()
    qkv_nonzero = all(inj.to_q.weight.grad.abs().max().item() > 0 for inj in injected.injectors)
    geo_nonzero = any(p.grad is not None and p.grad.abs().max().item() > 0
                      for p in injected.geo_tokenizer.parameters())
    print(f'[Test 3] with gate=0.5: QKV grads nonzero: {qkv_nonzero}, GeoTokenizer grads nonzero: {geo_nonzero}')
    assert qkv_nonzero and geo_nonzero
    with torch.no_grad():
        for inj in injected.injectors:
            inj.gate.zero_()
    results['ss/test3_gradients'] = 'PASS (gate grads nonzero @gate=0; QKV/GeoTokenizer grads nonzero @gate=0.5)'

    # --- Test 4: proxy_latent=None path ---
    injected.zero_grad(set_to_none=True)
    with torch.no_grad():
        y_none = injected(x, t, cond, proxy_latent=None)
    max_diff_none = (y_none - y_base).abs().max().item()
    print(f'[Test 4] proxy_latent=None forward OK, max|delta| vs base = {max_diff_none:.3e}')
    assert max_diff_none < 1e-3
    results['ss/test4_cfg_drop'] = f'PASS (max|delta|={max_diff_none:.3e})'

    del injected
    free()


def test_slat():
    print('\n' + '=' * 80)
    print('Shape flow: InjectedSLatFlowModel')
    print('=' * 80)
    x, t, cond, proxy_latent = make_slat_inputs()

    # --- Test 1: gate=0 equivalence against bare base model ---
    bare = models.from_pretrained(SHAPE_FLOW).cuda().eval()
    with torch.no_grad():
        y_base = bare(x, t, cond)
    del bare
    free()

    injected = models.InjectedSLatFlowModel(pretrained_base=SHAPE_FLOW).cuda()
    injected.base.eval()
    with torch.no_grad():
        y_inj = injected(x, t, cond, proxy_latent=proxy_latent)
    max_diff = (y_inj.feats - y_base.feats).abs().max().item()
    print(f'[Test 1] gate=0 equivalence: max|delta| = {max_diff:.3e}')
    assert max_diff < 1e-3, f'Equivalence failed: {max_diff}'
    results['slat/test1_equivalence'] = f'PASS (max|delta|={max_diff:.3e})'

    # --- Test 2: parameter stats ---
    trainable, frozen, base_all_frozen = param_stats(injected)
    print(f'[Test 2] trainable params: {trainable:,} ({trainable/1e6:.2f}M), '
          f'frozen params: {frozen:,} ({frozen/1e6:.2f}M), base frozen: {base_all_frozen}')
    assert trainable < PARAM_BUDGET, f'Trainable params exceed budget: {trainable}'
    assert base_all_frozen, 'Base model has trainable params'
    results['slat/test2_params'] = f'PASS (trainable={trainable/1e6:.2f}M < 80M, base frozen)'

    # --- Test 3: gradient flow ---
    pred = injected(x, t, cond, proxy_latent=proxy_latent)
    loss = pred.feats.float().sum()
    loss.backward()
    n_gate_grad = 0
    for i, inj in enumerate(injected.injectors):
        assert inj.gate.grad is not None, f'injector {i} gate grad is None'
        assert inj.to_q.weight.grad is not None, f'injector {i} to_q grad is None'
        assert inj.to_kv.weight.grad is not None, f'injector {i} to_kv grad is None'
        assert inj.to_out.weight.grad is not None, f'injector {i} to_out grad is None'
        if inj.gate.grad.abs().item() > 0:
            n_gate_grad += 1
    geo_grads = [p.grad is not None for p in injected.geo_tokenizer.parameters()]
    print(f'[Test 3] gate grads non-None: 30/30, nonzero: {n_gate_grad}/30; '
          f'GeoTokenizer grads non-None: {sum(geo_grads)}/{len(geo_grads)}')
    assert n_gate_grad == len(injected.injectors), 'some gate grads are zero'
    assert all(geo_grads), 'GeoTokenizer params missing grads'
    injected.zero_grad(set_to_none=True)
    with torch.no_grad():
        for inj in injected.injectors:
            inj.gate.fill_(0.5)
    pred = injected(x, t, cond, proxy_latent=proxy_latent)
    pred.feats.float().sum().backward()
    qkv_nonzero = all(inj.to_q.weight.grad.abs().max().item() > 0 for inj in injected.injectors)
    geo_nonzero = any(p.grad is not None and p.grad.abs().max().item() > 0
                      for p in injected.geo_tokenizer.parameters())
    print(f'[Test 3] with gate=0.5: QKV grads nonzero: {qkv_nonzero}, GeoTokenizer grads nonzero: {geo_nonzero}')
    assert qkv_nonzero and geo_nonzero
    with torch.no_grad():
        for inj in injected.injectors:
            inj.gate.zero_()
    results['slat/test3_gradients'] = 'PASS (gate grads nonzero @gate=0; QKV/GeoTokenizer grads nonzero @gate=0.5)'

    # --- Test 4: proxy_latent=None path ---
    injected.zero_grad(set_to_none=True)
    with torch.no_grad():
        y_none = injected(x, t, cond, proxy_latent=None)
    max_diff_none = (y_none.feats - y_base.feats).abs().max().item()
    print(f'[Test 4] proxy_latent=None forward OK, max|delta| vs base = {max_diff_none:.3e}')
    assert max_diff_none < 1e-3
    results['slat/test4_cfg_drop'] = f'PASS (max|delta|={max_diff_none:.3e})'

    del injected
    free()


if __name__ == '__main__':
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    ok = True
    for fn in [test_ss, test_slat]:
        try:
            fn()
        except Exception:
            traceback.print_exc()
            ok = False

    print('\n' + '=' * 80)
    print('SUMMARY')
    print('=' * 80)
    for k, v in results.items():
        print(f'{k:<28}{v}')
    if not ok:
        print('\nSOME TESTS FAILED')
        sys.exit(1)
    print('\nALL TESTS PASSED')
