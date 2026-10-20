// Minimal DOM/player mock only; this does not test browser audio decoding/playback.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const source=fs.readFileSync(__dirname+'/player.js','utf8');
function deferred(){let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b});return {promise,resolve,reject};}
const tick=()=>new Promise(r=>setImmediate(r));
async function run(){
 const events=[],requests=[],urls=[],revoked=[];
 const player={src:'',pause(){events.push('pause')},removeAttribute(n){assert.equal(n,'src');this.src='';events.push('clear')},load(){events.push('load')},async play(){events.push('play:'+this.src);if(this.fail)throw Error('autoplay blocked')}};
 const current={textContent:''},stop={},buttons=[{dataset:{file:'a.wav',name:'A'}},{dataset:{file:'b.wav',name:'B'}}],handlers={};
 const ctx={AbortController,document:{getElementById:id=>({player,current,stop})[id],querySelectorAll:()=>buttons},window:{addEventListener:(n,cb)=>handlers[n]=cb},URL:{createObjectURL:b=>{urls.push(b);return 'blob:'+b.id},revokeObjectURL:u=>revoked.push(u)},fetch:(file,opt)=>{const d=deferred();requests.push({file,opt,d});events.push('fetch:'+file);return d.promise}};
 vm.createContext(ctx);vm.runInContext(source,ctx);
 // Both header and blob must finish before play; stale body cannot replace new selection.
 const first=buttons[0].onclick();assert.deepEqual(events.slice(0,4),['pause','clear','load','fetch:a.wav']);
 const bodyA=deferred();requests[0].d.resolve({ok:true,blob:()=>bodyA.promise});await tick();assert.equal(urls.length,0);
 const second=buttons[1].onclick();assert.equal(requests[0].opt.signal.aborted,true);
 const bodyB=deferred();requests[1].d.resolve({ok:true,blob:()=>bodyB.promise});await tick();assert.equal(player.src,'');
 bodyB.resolve({id:'B'});await second;assert.equal(player.src,'blob:B');
 bodyA.resolve({id:'A'});await first;assert.equal(player.src,'blob:B');assert.deepEqual(urls.map(b=>b.id),['B']);
 // Stop clears source and revokes URL; stale fetch headers after stop cannot restart audio.
 stop.onclick();assert.equal(player.src,'');assert.deepEqual(revoked,['blob:B']);
 const third=buttons[0].onclick();stop.onclick();requests[2].d.resolve({ok:true,blob:async()=>({id:'stale'})});await third;assert.equal(player.src,'');
 // Old HTTP failure cannot overwrite the current status.
 const fourth=buttons[0].onclick(),fifth=buttons[1].onclick();requests[3].d.resolve({ok:false,status:500});await fourth;
 assert.equal(current.textContent,'正在载入：B');
 requests[4].d.resolve({ok:true,blob:async()=>({id:'B2'})});player.fail=true;await fifth;
 assert.ok(current.textContent.includes('未自动播放'));assert.equal(player.src,'blob:B2');
 handlers.pagehide();assert.equal(player.src,'');assert.ok(revoked.includes('blob:B2'));
 // Stale play rejection is excluded after switching and cannot overwrite new status.
 player.fail=false;const pending=deferred();player.play=()=>pending.promise;
 const sixth=buttons[0].onclick();requests[5].d.resolve({ok:true,blob:async()=>({id:'A2'})});await tick();
 const seventh=buttons[1].onclick();pending.reject(Error('late old play failure'));await sixth;
 assert.equal(current.textContent,'正在载入：B');stop.onclick();requests[6].d.reject(Error('aborted'));await seventh;
 console.log(JSON.stringify({status:'passed',scope:'minimal DOM mock; no actual browser playback',checks:['stop before switch','await fetch body','stale blob excluded','abort pending fetch','stop clears and revokes URL','stale HTTP failure excluded','autoplay rejection fallback','pagehide cleanup','stale play rejection excluded']}));
}
run().catch(e=>{console.error(e);process.exitCode=1});
