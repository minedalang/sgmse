'use strict';
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
const path=require('path');
const deferred=()=>{let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b});return {promise,resolve,reject};};
const flush=async()=>{for(let i=0;i<8;i++)await Promise.resolve();};
function setup(file){
 const elements={},audios=[],frames=new Map(),resumes=[];let nextFrame=0,delayPlay=false;
 function element(){return {textContent:'',checked:false,value:'1',append(){},addEventListener(){}};}
 for(const id of ['status','playing','links','loop','gain','equal','stop'])elements[id]=element();
 const document={getElementById:id=>elements[id],querySelectorAll:()=>[],body:{append(){}},createTextNode:s=>s,createElement:tag=>{
  if(tag!=='audio')return element();
  const listeners={};const p={readyState:0,currentTime:0,paused:true,playCalls:0,pending:[],addEventListener(n,f){(listeners[n]??=new Set()).add(f)},removeEventListener(n,f){listeners[n]?.delete(f)},emit(n){if(n==='loadedmetadata')this.readyState=1;for(const f of [...(listeners[n]||[])])f()},pause(){this.paused=true},play(){this.playCalls++;this.paused=false;if(delayPlay){const d=deferred();this.pending.push(d);return d.promise;}return Promise.resolve();}};audios.push(p);return p;
 }};
 class Context{constructor(){this.destination={}}resume(){const d=deferred();resumes.push(d);return d.promise}createMediaElementSource(){return {connect(){return {connect(){}}}}}createGain(){return {gain:{value:1},connect(){}}}}
 const c=vm.createContext({document,window:{AudioContext:Context,addEventListener(){}},requestAnimationFrame:f=>{frames.set(++nextFrame,f);return nextFrame},cancelAnimationFrame:id=>frames.delete(id)});
 vm.runInContext(fs.readFileSync(file,'utf8').match(/<script>([\s\S]*?)<\/script>/)[1],c);
 return {audios,frames,resumes,elements,run:s=>vm.runInContext(s,c),delayPlay:()=>{delayPlay=true}};
}
const dir=__dirname;
const tests={
 async resume_reordered(file){const t=setup(file);const a=t.run("play('clean',0,2,'first')");const b=t.run("play('off',0,2,'second')");t.audios.forEach(p=>p.readyState=1);t.resumes[1].resolve();await flush();t.resumes[0].resolve();await Promise.all([a,b]);assert.equal(t.audios[0].playCalls,0,'stale clean must never start');assert.equal(t.audios[1].paused,false);assert.equal(t.frames.size,1);},
 async metadata_reordered(file){const t=setup(file);const a=t.run("play('clean',0,2,'first')");t.resumes[0].resolve();await flush();const b=t.run("play('off',0,2,'second')");t.resumes[1].resolve();await flush();t.audios[1].emit('loadedmetadata');await flush();t.audios[0].emit('loadedmetadata');await Promise.all([a,b]);assert.equal(t.audios[0].playCalls,0);assert.equal(t.audios[1].paused,false);assert.equal(t.frames.size,1);},
 async stop_during_metadata(file){const t=setup(file);const a=t.run("play('clean',0,2,'first')");t.resumes[0].resolve();await flush();t.run('stop()');t.audios[0].emit('loadedmetadata');await a;assert.equal(t.audios[0].playCalls,0);assert.equal(t.frames.size,0);},
 async stop_during_resume(file){const t=setup(file);t.audios[0].readyState=1;const a=t.run("play('clean',0,2,'first')");t.run('stop()');t.resumes[0].resolve();await a;assert.equal(t.audios[0].playCalls,0);},
 async stale_play_rejection(file){const t=setup(file);t.delayPlay();t.audios.forEach(p=>p.readyState=1);const a=t.run("play('clean',0,2,'first')");t.resumes[0].resolve();await flush();const b=t.run("play('off',0,2,'second')");t.resumes[1].resolve();await flush();t.audios[1].pending[0].resolve();await b;t.audios[0].pending[0].reject(new Error('cancelled older request'));await a;assert.equal(t.audios[1].paused,false,'old rejection must not stop latest');assert.equal(t.frames.size,1);assert.match(t.elements.playing.textContent,/off.*second|second.*off/);},
 async stale_play_resolution(file){const t=setup(file);t.delayPlay();t.audios.forEach(p=>p.readyState=1);const a=t.run("play('clean',0,2,'first')");t.resumes[0].resolve();await flush();const b=t.run("play('off',0,2,'second')");t.resumes[1].resolve();await flush();t.audios[1].pending[0].resolve();await b;t.audios[0].pending[0].resolve();await a;assert.equal(t.audios[0].paused,true);assert.equal(t.audios[1].paused,false);assert.equal(t.frames.size,1);},
 async stop_during_play(file){const t=setup(file);t.delayPlay();t.audios[0].readyState=1;const a=t.run("play('clean',0,2,'first')");t.resumes[0].resolve();await flush();t.run('stop()');t.audios[0].pending[0].resolve();await a;assert.equal(t.audios[0].paused,true);assert.equal(t.frames.size,0);}
};
(async()=>{let errors=0;for(const file of ['listening_before_playback_fix_20261018.html','listening.html'])for(const [name,test]of Object.entries(tests)){let result='PASS',message='';try{await test(path.join(dir,file));}catch(e){result='FAIL';message=e.message;if(file==='listening.html')errors++;}console.log(JSON.stringify({file,test:name,result,message}));}process.exitCode=errors?1:0;})();
