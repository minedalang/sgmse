"""Experiment-local OFF replay with an explicit frozen noisy normalization."""
from functools import partial
import hashlib
import torch
from sgmse.util.other import pad_spec


def tensor_hash(x):
    return hashlib.sha256(x.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


@torch.inference_mode()
def fixed_norm_off(model, anchor, condition, refscale):
    assert model.train_type == 'mode2' and model.mode2_include_noisy_main
    assert model.sde.__class__.__name__ == 'OUVESDE' and model.sde.sampler_type == 'pc'
    assert not model.dnn.cross_attn and not model.dnn.film_modulate
    assert model.dnn.selector_type == 'none' and not model.residual_refine_enable
    assert anchor.shape == condition.shape == (32000,) and refscale > 0
    y = anchor[None, :].to(model.device) / refscale
    Y = pad_spec(torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0))
    c = condition[None, :].to(model.device) / refscale
    C = pad_spec(torch.unsqueeze(model._forward_transform(model._stft(c)), 0))
    regeneration = model._resolve_regeneration_strength(None, Y, C)
    meta = dict(normalized_anchor_sha256=tensor_hash(Y),
                normalized_condition_sha256=tensor_hash(C),
                cpu_rng_sha256=tensor_hash(torch.get_rng_state()),
                cuda_rng_sha256=tensor_hash(torch.cuda.get_rng_state()),
                regeneration_strength=None if regeneration is None else regeneration.detach().cpu().tolist(),
                condition_normalized_peak=float(c.abs().max().item()), trace_steps=0)
    def trace(state):
        if state['step_index'] == 0:
            meta['initial_xT_sha256'] = tensor_hash(state['x_t'])
        meta['trace_steps'] += 1
    sampler = model.get_pc_sampler('reverse_diffusion', 'none', Y.cuda(),
        score_fn=partial(model, y_cond=C), N=30, corrector_steps=1, snr=.5,
        intermediate=False, regeneration_strength=regeneration,
        anchor_calibrator=None, score_calibrator=None, trace_sink=trace)
    sample, nfe = sampler()
    result = (model.to_audio(sample.squeeze(), 32000) * refscale).squeeze().cpu().numpy()
    assert nfe == 30 and meta['trace_steps'] == 30
    meta['nfe'] = int(nfe)
    meta['outside_native_norm_range'] = meta['condition_normalized_peak'] > 1.0
    return result, meta
