"""Shared single-step native DSM/CPEL core; no execution on import."""
import copy
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import random
import os
import time
import numpy as np
import torch
from editing import crop_pair,edit_component,spectrum_mask
from loss_helpers import native_dsm,pair_loss
from gradient_budget import compose_gradients
import resume_io as rio


def canonical(state):
    h=hashlib.sha256()
    for n in sorted(state):
        t=state[n].detach().cpu().contiguous();h.update((n+str(tuple(t.shape))+str(t.dtype)).encode());h.update(t.numpy().tobytes())
    return h.hexdigest()


def cpu_copy(obj):
    if torch.is_tensor(obj):return obj.detach().cpu().clone()
    if isinstance(obj,Mapping):return {k:cpu_copy(v) for k,v in obj.items()}
    if type(obj) is list:return [cpu_copy(v) for v in obj]
    if type(obj) is tuple:return tuple(cpu_copy(v) for v in obj)
    return copy.deepcopy(obj)


class Core:
    def __init__(self,cfg,arm,registry,call_sink=None):
        from sgmse.joint_model import JointSemambaSgmseModel
        from sgmse.joint_data_module import JointWaveScpDataModule
        from sgmse.backbones.ncsnpp_utils.layerspp import GaussianFourierProjection
        self.cfg=cfg;self.arm=arm;self.registry=registry;self.device=torch.device(cfg['device']);self.physical_attempt=str(os.getpid())+'-'+str(time.time_ns())
        self.calls=dict(source_loads=0,pcm_reads=0,frontend=0,score=0,reverse=0,optimizer=0,public=0)
        if call_sink is not None:call_sink['physical_calls_this_process']=self.calls
        random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
        torch.cuda.set_device(self.device)
        self.calls['source_loads']+=1
        self.model=JointSemambaSgmseModel.load_from_checkpoint(cfg['source']['checkpoint']['path'],map_location='cpu',strict=True,
            weights_only=False,data_module_cls=JointWaveScpDataModule,frontend_ckpt=None)
        m=self.model;m.eval(no_ema=False)
        rio.require(m.train_type=='mode2' and type(m.sde).__name__=='OUVESDE' and m.frontend_name=='storm','Source topology mismatch')
        rio.require(m.loss_type=='score_matching' and m.loss_weighting=='sigma^2' and not m.residual_refine_enable,'Source loss/refinement mismatch')
        rio.require(not getattr(m.sde,'residual_anchor_enabled',False),'Residual anchor forbidden')
        for n,p in m.named_parameters():p.requires_grad_(n.startswith('dnn.'))
        self.intrinsic=[]
        for n,mod in m.dnn.named_modules():
            if isinstance(mod,GaussianFourierProjection):
                mod.W.requires_grad_(False);self.intrinsic.append(n+'.W' if n else 'W')
        rio.require(sorted(n for n,p in m.dnn.named_parameters() if not p.requires_grad)==sorted(self.intrinsic),'Unexpected fixed parameters')
        m.rrsd_freeze_base=False;m.rrsd_freeze_router=True;m.rrsd_disable_aux_loss=True;m.frontend_trainable=False
        m.to(self.device);m.dnn.eval();m.frontend.eval()  # Never model.train/eval again after injection.
        dm=m.data_module
        rio.require(m.sr==16000 and dm.n_fft==510 and dm.hop_length==128 and dm.num_frames==256 and dm.transform_type=='exponent'
            and dm.spec_abs_exponent==.5 and dm.spec_factor==.15,'Source geometry mismatch')
        self.params=[p for p in m.dnn.parameters() if p.requires_grad]
        rio.require(bool(self.params) and all(p.dtype==torch.float32 for p in self.params),'Parameter dtype/scope mismatch')
        self.optimizer=torch.optim.Adam(self.params,lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=0.)
        self.ema={n:p.detach().clone() for n,p in m.dnn.named_parameters() if p.requires_grad}
        self.generators={'shuffle_crop_parent':torch.Generator().manual_seed(cfg['seed']),
            'native_pair_noise_time':torch.Generator(device=self.device).manual_seed(cfg['seed'])}
        self.order=[];self.step=0;self.frozen=self.frozen_digest()
        self.signature=self.parameter_digest()
        self.hook=m.dnn.register_forward_hook(self._score_hook)
        # Source construction may consume global RNG; canonical training boundary is reset identically.
        random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    def parameter_digest(self):
        return hashlib.sha256(json.dumps([(n,list(p.shape),str(p.dtype),p.requires_grad) for n,p in self.model.dnn.named_parameters()],separators=(',',':')).encode()).hexdigest()
    def _score_hook(self,module,args,out):self.calls['score']+=int(args[0].shape[0])
    def frozen_digest(self):
        return dict(frontend=canonical(self.model.frontend.state_dict()),intrinsic=canonical({n:self.model.dnn.state_dict()[n] for n in self.intrinsic}))
    def check_frozen(self):rio.require(self.frozen_digest()==self.frozen,'Frozen frontend/Fourier changed')
    def read(self,path):
        import soundfile as sf
        a,sr=sf.read(path,dtype='float32',always_2d=True);self.calls['pcm_reads']+=1
        rio.require(sr==16000 and a.shape[1]==1 and np.isfinite(a).all(),'Expected finite mono16k')
        return torch.from_numpy(a[:,0].copy()).reshape(1,-1)
    def spec(self,x):return self.model._sgmse_stft(x.to(self.device))
    def front(self,n):
        self.calls['frontend']+=int(n.shape[0])
        with torch.no_grad():return self.model.frontend.forward_transformed_spec(n).detach()
    def geometry(self,t):
        a,b=self.model.sde.weights(t);return a,b,self.model.sde._std(t).reshape(-1,1,1,1)
    def noise(self,z):return torch.randn(z.shape,device=self.device,dtype=z.dtype,generator=self.generators['native_pair_noise_time'])
    def update(self):
        cpu=self.generators['shuffle_crop_parent'];gpu=self.generators['native_pair_noise_time'];m=self.model
        if not self.order:self.order=torch.randperm(len(self.registry),generator=cpu).tolist()
        row=self.registry[self.order.pop()]
        s,x,start,factor=crop_pair(self.read(row['clean_path_loader']),self.read(row['wav_path']),32640,cpu,True)
        delta,center=edit_component(s,'hann');c1,c0,n1,n0=self.spec(s),self.spec(s-delta),self.spec(x),self.spec(x-delta)
        f1,f0=self.front(n1),self.front(n0);p=spectrum_mask(c1,center,m.data_module.hop_length)
        t=.03+.97*torch.rand(1,device=self.device,generator=gpu);a,b,sig=self.geometry(t);eps=self.noise(c1)
        q1=m(a*c1+b*f1+sig*eps,f1,t,y_cond=n1);q0=m(a*c0+b*f0+sig*eps,f0,t,y_cond=n0)
        native=(native_dsm(q1,eps,sig)+native_dsm(q0,eps,sig))/2
        ta=.3+.4*torch.rand(1,device=self.device,generator=gpu);a,b,sig=self.geometry(ta);parent=int(torch.randint(2,(),generator=cpu))
        cp,anchor=(c0,f0) if parent==0 else (c1,f1);u=(a*cp+b*anchor+sig*self.noise(cp)).detach()
        qa1,qa0=m(u,anchor,ta,y_cond=n1),m(u,anchor,ta,y_cond=n0)
        pair,stats=pair_loss(qa1,qa0,c1-c0,p,a,sig)
        rio.require(bool(torch.isfinite(native)) and bool(torch.isfinite(pair)),'Nonfinite objective')
        self.optimizer.zero_grad(set_to_none=True)
        gm=torch.autograd.grad(native,self.params,retain_graph=True,allow_unused=True);self.calls['reverse']+=1
        gp=torch.autograd.grad(pair,self.params,allow_unused=True);self.calls['reverse']+=1
        combined,budget=compose_gradients(gm,gp,stats['valid'],self.arm=='D')
        for param,grad in zip(self.params,combined):param.grad=grad
        grad=float(torch.nn.utils.clip_grad_norm_(self.params,1.,error_if_nonfinite=True))
        self.optimizer.step();self.calls['optimizer']+=1
        with torch.no_grad():
            for n,param in m.dnn.named_parameters():
                if n in self.ema:self.ema[n].mul_(.999).add_(param,alpha=.001)
        self.step+=1
        return dict(step=self.step,utt=row['utt'],crop_start=start,normfactor=factor,center=center,parent=parent,t=float(t),aux_t=float(ta),
            native=float(native.detach()),pair=float(pair.detach()),pair_stats=stats,gradient_budget=budget,grad_preclip=grad,common_clean_aux_computed=False)
    def snapshot(self,identity,commit):
        torch.cuda.synchronize(self.device);self.check_frozen()
        return dict(schema=rio.SCHEMA,identity=identity,completed_step=self.step,boundary='optimizer_ema_log_committed',
            raw=cpu_copy(self.model.dnn.state_dict()),ema=cpu_copy(self.ema),ema_decay=.999,ema_updates=self.step,
            optimizer=cpu_copy(self.optimizer.state_dict()),rng=rio.capture_rng(self.generators,{str(self.device.index):torch.cuda.get_rng_state(self.device)}),
            remaining_order=list(self.order),registry_count=len(self.registry),log_commit=commit)
    def restore(self,packet):
        rio.require(self.parameter_digest()==self.signature and [id(p) for p in self.params]==[id(p) for p in self.model.dnn.parameters() if p.requires_grad],'Actual model/optimizer parameter binding changed')
        rio.require(packet['registry_count']==len(self.registry),'Registry size differs')
        self.model.dnn.load_state_dict(packet['raw'],strict=True);self.optimizer.load_state_dict(packet['optimizer'])
        rio.require(set(packet['ema'])==set(self.ema),'EMA parameter identity mismatch')
        self.ema={n:t.to(self.device).clone() for n,t in packet['ema'].items()}
        for p in self.params:p.grad=None
        self.order=list(packet['remaining_order']);self.step=packet['completed_step'];self.check_frozen()
        rio.restore_rng(packet['rng'],self.generators,lambda states:torch.cuda.set_rng_state(states[str(self.device.index)],device=self.device))
    def close(self):self.hook.remove()
