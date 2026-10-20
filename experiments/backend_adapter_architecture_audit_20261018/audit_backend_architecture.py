#!/usr/bin/env python3
"""Read-only audit of adapter insertion candidates for the locked 80k Score model."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path('/data/coding/sgmse')
sys.path.insert(0, str(ROOT))
CKPT = ROOT / 'joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt'
OUT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    checkpoint = torch.load(CKPT, map_location='cpu', weights_only=False)
    state = checkpoint['state_dict']
    hparams = checkpoint['hyper_parameters']
    dnn = [(k, tuple(v.shape), int(v.numel())) for k, v in state.items()
           if k.startswith('dnn.') and hasattr(v, 'shape')]

    module_rows = []
    for idx in range(34):
        values = [(k, shape, n) for k, shape, n in dnn
                  if k.startswith(f'dnn.all_modules.{idx}.')]
        if values:
            module_rows.append({
                'all_modules_index': idx,
                'parameter_count': sum(n for _, _, n in values),
                'tensor_count': len(values),
                'shapes': [{'name': k, 'shape': list(shape), 'parameters': n}
                           for k, shape, n in values],
            })

    convs = []
    for k, shape, n in dnn:
        if k.startswith('dnn.all_modules.') and k.endswith('.weight') and len(shape) == 4:
            out_ch, in_ch, kh, kw = shape
            if kh > 1 or kw > 1:
                # This is only a parameter-budget estimate for a flattened LoRA factorization.
                rank = 4
                lora_params = rank * (in_ch * kh * kw + out_ch)
                convs.append({
                    'name': k,
                    'shape': list(shape),
                    'base_parameters': n,
                    'rank4_flattened_lora_parameters': lora_params,
                    'rank4_fraction_of_base': lora_params / n,
                })

    late = [x for x in convs if any(x['name'].startswith(f'dnn.all_modules.{i}.')
                                    for i in (31, 32, 33))]
    result = {
        'experiment_id': 'backend_adapter_architecture_audit_20261018',
        'status': 'audit_complete_no_training',
        'checkpoint': {
            'path': str(CKPT),
            'sha256': sha256(CKPT),
            'global_step': checkpoint.get('global_step'),
            'epoch': checkpoint.get('epoch'),
        },
        'model_hyperparameters': {
            key: hparams.get(key) for key in
            ('backbone', 'nf', 'ch_mult', 'num_res_blocks', 'attn_resolutions',
             'train_type', 'sde', 'loss_type', 'loss_weighting', 'network_scaling')
        },
        'dnn_parameter_count': sum(n for _, _, n in dnn),
        'dnn_tensor_count': len(dnn),
        'module_rows': module_rows,
        'spatial_conv_count': len(convs),
        'late_decoder_candidates': late,
        'candidate_summary': {
            'late_decoder_base_parameters': sum(x['base_parameters'] for x in late),
            'late_decoder_rank4_flattened_lora_parameters': sum(
                x['rank4_flattened_lora_parameters'] for x in late),
            'note': 'LoRA counts are a design estimate, not implemented parameters; decoder insertion and zero-init behavior still require a separate code change and smoke test.',
        },
        'data_or_checkpoint_modified': False,
    }
    (OUT / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in
                      ('dnn_parameter_count', 'dnn_tensor_count', 'spatial_conv_count', 'candidate_summary')}, indent=2))


if __name__ == '__main__':
    main()
