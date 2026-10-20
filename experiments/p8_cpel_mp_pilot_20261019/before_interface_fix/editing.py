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
