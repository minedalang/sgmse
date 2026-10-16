#!/usr/bin/env python3
"""Read-only Future/Perceptual Gradient Agreement diagnostic for frozen RRSD.

Launch this file (NOT joint_train.py) using the wrapper shell script.  It
monkey-patches one class method before executing joint_train.py, without
editing source or checkpoints. Diagnostic only, never train WavLM.
"""
import os
import sys
import runpy
import math
import torch
import torch.nn.functional as F

_WAVLM = None


def _encoder(device):
    global _WAVLM
    if _WAVLM is None:
        from transformers import WavLMModel
        model_id = os.environ.get('PG_MODEL_PATH', 'microsoft/wavlm-base-plus')
        offline = os.environ.get('PG_LOCAL_ONLY', '1') == '1'
        _WAVLM = WavLMModel.from_pretrained(model_id, local_files_only=offline)
        _WAVLM.eval().requires_grad_(False).to(device)
        print(f'[RRSD-PG-INIT] encoder={model_id} local_only={int(offline)} layer={os.environ.get("PG_LAYER", "3")}', flush=True)
    return _WAVLM


def _wave(model, spec, max_length):
    """Correct SGMSE complex inverse-transform -> iSTFT, differentiably."""
    length = (int(model.data_module.num_frames) - 1) * int(model.data_module.hop_length)
    xs = []
    for i in range(spec.shape[0]):
        wav = model.to_audio(spec[i].squeeze(0), length=length).reshape(-1)
        if wav.numel() >= max_length:
            wav = wav[:max_length]
        else:
            wav = F.pad(wav, (0, max_length - wav.numel()))
        xs.append(wav)
    x = torch.stack(xs, dim=0).float()
    x = x - x.mean(dim=-1, keepdim=True)
    x = x / (x.abs().amax(dim=-1, keepdim=True) + 1e-6)
    return x


def _feat(encoder, x, layer):
    hs = encoder(input_values=x, output_hidden_states=True, return_dict=True).hidden_states
    if layer >= len(hs):
        raise RuntimeError(f'WavLM layer {layer} unavailable; {len(hs)} hidden states')
    return hs[layer].float()


def _future_at(model, alpha, x_t, y_main, y_cond, t, score, sigma, w_x,
               front_dir, dt, target):
    interp = F.interpolate(alpha, size=front_dir.shape[-2:], mode='bilinear', align_corners=False)
    interp = interp.to(dtype=front_dir.real.dtype)
    score_route = score + w_x * interp * front_dir / sigma.square().clamp_min(1e-8)
    x_next, t_next = model._rrsd_reverse_mean_step(
        x=x_t, y_main=y_main, t=t, score=score_route, dt=dt)
    score_next = model(x_next, y_main, t_next, y_cond=y_cond)
    x0_next, _, _, _ = model._rrsd_score_to_x0(x_next, y_main, t_next, score_next)
    err = (x0_next - target).abs().float().square().flatten(1).mean(1)
    return x0_next, err


def _run_probe(model, *, target, x0_front, step_out, pred, batch_num, max_batches):
    score, x_t, y_main, t, sigma, wx, _, x0_base = model._score_step_geometry(step_out)
    y_cond = step_out.get('y_cond', None)
    if y_cond is None:
        y_cond = y_main
    t = t.detach().float().flatten()
    B = len(t)
    tmin = float(os.environ.get('PG_T_MIN', '0.05'))
    tmax = float(os.environ.get('PG_T_MAX', '0.50'))
    chosen = torch.nonzero((t >= tmin) & (t < tmax), as_tuple=False).flatten()
    if chosen.numel() == 0:
        print(f'[RRSD-PG-SKIP] batch={batch_num} t=' + ','.join(f'{v:.5f}' for v in t.tolist()), flush=True)
        return
    if int(getattr(model, 'sr', 16000)) != 16000:
        raise RuntimeError('This probe requires 16kHz audio for official WavLM')
    def sl(z):
        return z.detach().index_select(0, chosen)
    x_t, y_main, y_cond, sigma, wx, score = map(sl, (x_t,y_main,y_cond,sigma,wx,score))
    x0_base, x0_front, target = map(sl, (x0_base,x0_front,target))
    t = t.index_select(0,chosen)
    router_alpha = sl(pred['alpha_tile']).float()
    anchor_name = os.environ.get('PG_ANCHOR','zero')
    if anchor_name not in ('zero','router'):
        raise ValueError('PG_ANCHOR must be zero or router')
    anchor = torch.zeros_like(router_alpha) if anchor_name=='zero' else router_alpha
    alpha0 = anchor.detach().clone().requires_grad_(True)
    base_dt = float(model.rrsd_lookahead_dt)
    if base_dt <= 0:
        base_dt = (float(getattr(model.sde,'T',1.0))-float(model.t_eps)) / max(int(getattr(model.sde,'N',30))-1,1)
    dt = torch.minimum(torch.full_like(t,max(base_dt,1e-6)), (t-float(model.t_eps)).clamp_min(0.0))
    if (dt<=0).any():
        raise RuntimeError('t too near t_eps')

    wavlm = _encoder(x_t.device)
    layer = int(os.environ.get('PG_LAYER', '3'))
    maxlen = int(os.environ.get('PG_MAX_SAMPLES', '32000'))
    with torch.no_grad():
        clean_wave = _wave(model, target, maxlen)
        h_clean = _feat(wavlm,clean_wave,layer).detach()
    with torch.enable_grad():
        future_x0, err0 = _future_at(model, alpha0, x_t,y_main,y_cond,t,score,sigma,wx,
                                     x0_front-x0_base, dt, target)
        # Same future x0 for BOTH losses, matching original Future-grad time.
        loss_future_vec = err0 / err0.detach().clamp_min(float(model.rrsd_oracle_eps))
        h_future = _feat(wavlm, _wave(model,future_x0,maxlen), layer)
        loss_perc_vec = (h_future-h_clean).square().mean(dim=(1,2))
        gf = torch.autograd.grad(loss_future_vec.sum(), alpha0, retain_graph=True)[0].detach().float()
        gp = torch.autograd.grad(loss_perc_vec.sum(), alpha0)[0].detach().float()

    rel_tol = float(os.environ.get('PG_REL_TOL','0.10'))
    eps_f = float(os.environ.get('PG_F_MIN','1e-10'))
    eps_p = float(os.environ.get('PG_P_MIN','1e-10'))
    if not (torch.isfinite(gf).all() and torch.isfinite(gp).all()):
        raise RuntimeError('Nonfinite gradient; check WavLM/iSTFT and numerical precision')
    B2 = int(t.shape[0]); ntiles = gf[0].numel()
    af=router_alpha.flatten(1); ff=gf.flatten(1); pp=gp.flatten(1)
    tf=torch.maximum(ff.abs().mean(1)*rel_tol, torch.full_like(t, eps_f)).unsqueeze(1)
    tp=torch.maximum(pp.abs().mean(1)*rel_tol, torch.full_like(t, eps_p)).unsqueeze(1)
    fgood=ff < -tf; pgood=pp < -tp
    fbad=ff > tf; pbad=pp > tp
    conflict=fgood & pbad; agree=fgood & pgood; bothbad=fbad & pbad
    active = af > float(os.environ.get('PG_ACTIVE_ALPHA', '0.001'))
    def frac(z): return z.float().mean(1)
    # Mask perturbation: top-|gradient| tiles; never use as a training oracle.
    # Choose signed *scale-normalized* gradients so neither task dominates.
    sf=ff.abs()/(ff.abs().mean(1,keepdim=True)+1e-12)
    sp=pp.abs()/(pp.abs().mean(1,keepdim=True)+1e-12)
    priority = sf+sp
    topk = min(int(os.environ.get('PG_FD_TOPK','32')), ntiles)
    fd_mask=torch.zeros_like(ff).scatter_(1,priority.topk(topk,dim=1).indices,1.0)
    fd_eps=float(os.environ.get('PG_FD_EPS', '0.02'))
    fd_max=int(os.environ.get('PG_FD_BATCHES', '4'))
    fd_do=batch_num<=fd_max
    obs_f=obs_p=pred_f=pred_p=None
    if fd_do:
        with torch.no_grad():
            ap=(alpha0.detach()+fd_mask.view_as(alpha0)*fd_eps).clamp(0,1)
            xp, e_plus = _future_at(model,ap,x_t,y_main,y_cond,t,score,sigma,wx,
                                     x0_front-x0_base,dt,target)
            h_plus = _feat(wavlm,_wave(model,xp,maxlen),layer)
            l_plus = (h_plus-h_clean).square().mean(dim=(1,2))
            obs_f=(e_plus/err0.detach().clamp_min(float(model.rrsd_oracle_eps))-loss_future_vec.detach())
            obs_p=(l_plus-loss_perc_vec.detach())
            pred_f=fd_eps*(ff*fd_mask).sum(1)
            pred_p=fd_eps*(pp*fd_mask).sum(1)

    for i in range(B2):
        wconf = float((af[i]*conflict[i]).sum()/(af[i].sum()+1e-10))
        ac=active[i]; n_active=int(ac.sum().item())
        idx=int(chosen[i].item())
        line=(f'[RRSD-PG] batch={batch_num} sample={idx} anchor={anchor_name} t={t[i].item():.5f} '
              f'alpha_mean={af[i].mean().item():.8f} active={frac(active)[i].item():.5f} active_tiles={n_active} '
              f'gf_abs={ff[i].abs().mean().item():.5e} gp_abs={pp[i].abs().mean().item():.5e} '
              f'gf_thr={tf[i,0].item():.4e} gp_thr={tp[i,0].item():.4e} '
              f'fgood={frac(fgood)[i].item():.5f} fbad={frac(fbad)[i].item():.5f} '
              f'pgood={frac(pgood)[i].item():.5f} pbad={frac(pbad)[i].item():.5f} '
              f'conflict={frac(conflict)[i].item():.5f} agree={frac(agree)[i].item():.5f} '
              f'bothbad={frac(bothbad)[i].item():.5f} '
              f'conflict_on_active={((conflict[i]&ac).sum().float()/max(n_active,1)).item():.5f} '
              f'alpha_conflict_budget={wconf:.5f} '
              f'base_ssl_mse={loss_perc_vec[i].item():.6e} '
              f'base_future_mse={err0[i].item():.6e}')
        if fd_do:
            # Thresholds separate numerical noise from reliable finite differences.
            sf_ok=abs(obs_f[i].item())>float(os.environ.get('PG_FD_F_MIN','1e-6'))
            sp_ok=abs(obs_p[i].item())>float(os.environ.get('PG_FD_P_MIN','1e-7'))
            fc=sf_ok and obs_f[i].item()*pred_f[i].item()>0
            pc=sp_ok and obs_p[i].item()*pred_p[i].item()>0
            line+=(f' fd=1 fd_f_pred={pred_f[i].item():+.6e} fd_f_obs={obs_f[i].item():+.6e} '
                   f'fd_f_valid={int(sf_ok)} fd_f_sign_ok={int(fc)} '
                   f'fd_p_pred={pred_p[i].item():+.6e} fd_p_obs={obs_p[i].item():+.6e} '
                   f'fd_p_valid={int(sp_ok)} fd_p_sign_ok={int(pc)}')
        else:
            line+=' fd=0'
        print(line,flush=True)


def main():
    from sgmse.joint_model import JointFrontendSgmseModel
    original=JointFrontendSgmseModel._rrsd_future_grad_loss
    def wrapper(self, *, target, x0_front, step_out, pred):
        n=int(getattr(self,'_rrsd_pg_probe_n',0))+1
        limit=int(os.environ.get('PG_BATCHES','4'))
        if n<=limit:
            if not bool(getattr(self,'rrsd_freeze_base',False)):
                raise RuntimeError('PG diagnostic requires --rrsd_freeze_base')
            self._rrsd_pg_probe_n=n
            _run_probe(self,target=target,x0_front=x0_front,step_out=step_out,pred=pred,
                       batch_num=n,max_batches=limit)
            if n>=limit:
                print('[RRSD-PG-DONE] finished without updating/saving any checkpoint',flush=True)
                raise SystemExit(0)
        return original(self,target=target,x0_front=x0_front,step_out=step_out,pred=pred)
    JointFrontendSgmseModel._rrsd_future_grad_loss=wrapper
    script='joint_train.py'
    # Executing as __main__ preserves argument parsing and Lightning behavior.
    sys.argv=[script]+sys.argv[1:]
    runpy.run_path(script,run_name='__main__')


if __name__=='__main__':
    main()
