'use strict';
const fs=require('fs'),path=require('path'),http=require('http'),assert=require('assert/strict');
const {chromium}=require('/data/codex-state/tmp/audio-paircheck-browser-20261018/node_modules/playwright');
const root=path.resolve(__dirname,'..');
const server=http.createServer((req,res)=>{const p=path.resolve(root,decodeURIComponent(req.url.split('?')[0]).slice(1));if(!p.startsWith(root+'/')){res.writeHead(403);return res.end();}fs.readFile(p,(e,b)=>{if(e){res.writeHead(404);res.end();return;}res.setHeader('Content-Type',p.endsWith('.wav')?'audio/wav':'text/html');res.end(b);});});
const results=[];
const state=p=>p.evaluate(()=>({playing:[...document.querySelectorAll('audio')].filter(x=>!x.paused).map(x=>({src:x.getAttribute('src'),time:x.currentTime})),label:document.getElementById('playing')?.textContent||null}));
(async()=>{await new Promise(r=>server.listen(0,'127.0.0.1',r));const origin='http://127.0.0.1:'+server.address().port;
const browser=await chromium.launch({headless:true,args:['--no-sandbox','--autoplay-policy=no-user-gesture-required']});
try{
for(const filename of ['listening_before_playback_fix_20261018.html','listening.html']){
 for(const scenario of ['metadata_switch','metadata_stop','rapid_switch','window_B','play_pending_stop']){
  const page=await browser.newPage();let release;const held=new Promise(r=>release=r);let failed=false;
  try{
   if(scenario.startsWith('metadata'))await page.route('**/*_clean.wav',async route=>{await held;await route.continue();});
   if(scenario==='play_pending_stop')await page.addInitScript(()=>{const original=HTMLMediaElement.prototype.play;window.playResolvers=[];HTMLMediaElement.prototype.play=function(){const p=original.call(this);return p.then(()=>new Promise(r=>window.playResolvers.push(r)));};});
   await page.goto(origin+'/p8_10720_content_identity_audit_20261018/'+filename,{waitUntil:'domcontentloaded'});
   const group=page.locator('#context');
   if(scenario.startsWith('metadata')){
    await group.getByRole('button',{name:'播放 clean',exact:true}).click();await page.waitForTimeout(80);
    if(scenario==='metadata_switch')await group.getByRole('button',{name:'播放 off',exact:true}).click();else await page.locator('#stop').click();
    release();await page.waitForTimeout(250);
    const s=await state(page);if(scenario==='metadata_stop')assert.equal(s.playing.length,0);else{assert.equal(s.playing.length,1);assert.match(s.playing[0].src,/_off.wav$/);}results.push({filename,scenario,result:'PASS',state:s});
   }else if(scenario==='rapid_switch'){
    await page.evaluate(()=>{for(const n of ['clean','off','router'])[...document.querySelectorAll('#context button')].find(b=>b.textContent==='播放 '+n).click();});await page.waitForTimeout(250);const s=await state(page);assert.equal(s.playing.length,1);assert.match(s.playing[0].src,/_router.wav$/);if(filename==='listening.html')assert.match(s.label,/router.*context_router.wav/);results.push({filename,scenario,result:'PASS',state:s});
   }else if(scenario==='window_B'){
    await page.locator('#B').getByRole('button',{name:'播放 off',exact:true}).click();await page.waitForFunction(()=>[...document.querySelectorAll('audio')].some(p=>!p.paused));const s=await state(page);assert.equal(s.playing.length,1);assert(s.playing[0].time>=.45&&s.playing[0].time<.8);results.push({filename,scenario,result:'PASS',state:s});
   }else{
    await group.getByRole('button',{name:'播放 clean',exact:true}).click();await page.waitForFunction(()=>window.playResolvers.length>0);await page.locator('#stop').click();await page.evaluate(()=>window.playResolvers.forEach(r=>r()));await page.waitForTimeout(150);const s=await state(page);assert.equal(s.playing.length,0);results.push({filename,scenario,result:'PASS',state:s});
   }
  }catch(e){failed=true;results.push({filename,scenario,result:'FAIL',error:e.message,state:await state(page)});if(filename==='listening.html')process.exitCode=1;}
  finally{release();await page.close();}
 }
}
for(const filename of ['listening_before_single_player_fix_20261018.html','listening.html']){
 const page=await browser.newPage();try{
 await page.goto(origin+'/p8_e1_candidate_listening_aligned_20261020/'+filename,{waitUntil:'domcontentloaded'});
 await page.evaluate(async()=>{const players=[...document.querySelectorAll('audio')].filter(p=>p.src.includes('10720'));await players[0].play();await players[3].play();});
 const s=await state(page);assert.equal(s.playing.length,1);assert.match(s.playing[0].src,/_off.wav$/);results.push({filename,scenario:'native_single_player',result:'PASS',state:s});
 }catch(e){results.push({filename,scenario:'native_single_player',result:'FAIL',error:e.message,state:await state(page)});if(filename==='listening.html')process.exitCode=1;}finally{await page.close();}
}
}finally{await browser.close();server.close();}
const out=path.join(__dirname,'playback_browser_results_20261018.json');fs.writeFileSync(out,JSON.stringify(results,null,2)+'\n',{flag:'wx'});console.log(JSON.stringify(results,null,2));
})().catch(e=>{console.error(e);server.close();process.exitCode=1});
