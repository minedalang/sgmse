"""Deterministic waveform edits and compressed-spectrum temporal soft masks."""
import math
import torch


def bandpass(x, sr=16000):
    freq = torch.fft.rfftfreq(x.shape[-1], d=1/sr, device=x.device)
    gain = torch.zeros_like(freq)
    rise = (freq >= 1500) & (freq < 2500)
    fall = (freq > 6500) & (freq < 7500)
    gain[rise] = .5-.5*torch.cos(math.pi*(freq[rise]-1500)/1000)
    gain[(freq >= 2500) & (freq <= 6500)] = 1
    gain[fall] = .5+.5*torch.cos(math.pi*(freq[fall]-6500)/1000)
    return torch.fft.irfft(torch.fft.rfft(x)*gain, n=x.shape[-1])


def temporal(n, center, width, kind, device, dtype, sr=16000):
    distance = (torch.arange(n, device=device, dtype=dtype)-center).abs()/sr
    half = width/2
    if kind == 'hann':
        return torch.where(distance < half, .5+.5*torch.cos(math.pi*distance/half), 0.)
    if kind == 'tukey':
        edge = .010
        inner = half-edge
        return torch.where(distance <= inner, 1., torch.where(distance < half,
            .5+.5*torch.cos(math.pi*(distance-inner)/edge), 0.))
    raise ValueError(kind)


def edit_component(clean, kind='hann', center=None, sr=16000):
    filtered = bandpass(clean, sr)
    n = clean.shape[-1]
    if center is None:
        # 80 ms moving energy, centers at least .25 s from each crop boundary.
        win = int(round(.080*sr))
        power = torch.nn.functional.pad(filtered.square().reshape(-1), (win//2, win//2))
        integral = torch.cat([power.new_zeros(1), power.cumsum(0)])
        energy = (integral[win:]-integral[:-win])[:n]
        margin = int(round(.25*sr))
        if n <= 2*margin:
            raise ValueError('Crop shorter than event selection margins')
        center = int(energy[margin:n-margin].argmax())+margin
    window = temporal(n, center, .080, kind, clean.device, clean.dtype, sr)
    return .5*filtered*window, center


def spectrum_mask(spec, center, hop, sr=16000):
    # center=True STFT frame j is centered at sample j*hop.
    location = torch.arange(spec.shape[-1], device=spec.device,
                            dtype=spec.real.dtype)*hop
    distance = (location-center).abs()/sr
    p = torch.where(distance < .096, .5+.5*torch.cos(math.pi*distance/.096), 0.)
    return p.reshape(*([1]*(spec.ndim-1)), -1)


def crop_pair(clean, noisy, length, generator=None, training=False):
    if clean.shape != noisy.shape:
        raise ValueError('Clean/Noisy lengths must agree; no silent min-length alignment')
    n = clean.shape[-1]
    if n < length:
        pad = length-n
        clean = torch.nn.functional.pad(clean, (pad//2, pad-pad//2))
        noisy = torch.nn.functional.pad(noisy, (pad//2, pad-pad//2))
        start = -(pad//2)
    else:
        start = int(torch.randint(n-length+1, (), generator=generator)) if training else (n-length)//2
        clean, noisy = clean[..., start:start+length], noisy[..., start:start+length]
    factor = noisy.abs().max().clamp_min(1e-12)
    return clean/factor, noisy/factor, start, float(factor)


def pcm16_nearest_no_clip(wave):
    """Physical-amplitude fixed PCM16 nearest ties-to-even, with no saturation."""
    physical=wave.to(torch.float64)
    codes=torch.round(physical*32768.)
    finite=bool(torch.isfinite(codes).all())
    outside=int(((codes < -32768) | (codes > 32767)).sum()) if finite else None
    info=dict(quantizer='PCM16 nearest ties-to-even float64, q=1/32768',
        finite=finite,out_of_code_range_samples=outside,clipped=False,
        endpoint_code_samples=int(((codes==-32768)|(codes==32767)).sum()) if finite else None,
        valid=finite and outside==0)
    if not info['valid']:return None,info
    return codes/32768.,info


def codec_additivity(s,x,n,subtypes):
    """Engineering compatibility bounds; never establish the processing chain."""
    result=dict(subtypes=subtypes,processing_chain_verified=False,
        interpretation='codec compatibility only; not noise purity, a true negative or FPR')
    if not s.shape==x.shape==n.shape:
        return dict(**result,status='unknown_length_mismatch',strict_pass=None,
                    nearest_compatible=None,general_compatible=None)
    err=(x.to(torch.float64)-s.to(torch.float64)-n.to(torch.float64))
    rms=float(err.square().mean().sqrt()/x.to(torch.float64).square().mean().sqrt().clamp_min(1e-12))
    result.update(strict_relative_rms=rms,strict_pass=rms<=.001,
        max_absolute_residual=float(err.abs().max()))
    # Only known integer PCM formats get q bounds. Other codecs are unknown,
    # not silently assigned a fictitious quantization step.
    bits={'PCM_16':16,'PCM_24':24,'PCM_32':32}
    if any(v not in bits for v in subtypes):
        return dict(**result,status='unknown_codec',nearest_compatible=None,general_compatible=None)
    qs=[2.**(1-bits[v]) for v in subtypes]
    endpoints=[]
    for w,q in zip((s,x,n),qs):
        endpoints.append(int(((w<=-1.) | (w>=1-q)).sum()))
    qsum=sum(qs);nearest=.5*qsum;general=qsum
    result.update(quantization_steps=qs,endpoint_samples=endpoints,
        nearest_bound=nearest,general_bound=general,
        nearest_compatible=bool((err.abs()<=nearest+1e-12).all()),
        general_compatible=bool((err.abs()<=general+1e-12).all()),
        nearest_exceed_samples=int((err.abs()>nearest+1e-12).sum()),
        general_exceed_samples=int((err.abs()>general+1e-12).sum()))
    result['status']='codec_compatible' if result['general_compatible'] and not any(endpoints) else 'unknown_codec_or_clipping'
    return result
