'use strict';const assert=require('assert'),fs=require('fs'),api=require('./player.js');const checks=[];function ok(x,label){assert(x,label);checks.push(label);}
ok(!api.timingInfo(.199999).target&&api.timingInfo(.2).target&&api.timingInfo(.279999).target&&!api.timingInfo(.28).target,'exact 200inclusive280exclusive target boundaries');
ok(api.timingInfo(-1).ms===0&&api.timingInfo(1).ms===480&&api.timingInfo(.24).percent===50,'clamped readonly480ms timeline position');
const audio={currentTime:0},draws=[],pending=new Map();let counter=0;const ctl=api.timingController(audio,x=>draws.push(x),fn=>{pending.set(++counter,fn);return counter;},id=>pending.delete(id));
ctl.start();let id=counter,frame=pending.get(id);pending.delete(id);audio.currentTime=.22;frame();ok(draws.at(-1).target&&draws.at(-1).ms===220,'RAF follows audio.currentTime not wallclock');
const stale=pending.get(counter);ctl.stop();audio.currentTime=0;ctl.start();const before=draws.length,jobs=pending.size;stale();ok(draws.length===before&&pending.size===jobs,'stale RAF cannot paint or schedule after new render generation');
ctl.stop();audio.currentTime=0;ctl.paint();ok(draws.at(-1).ms===0&&pending.size===0,'render pause reset cancels prior frame');
ctl.start();audio.currentTime=.48;ctl.stop();ok(draws.at(-1).ms===480&&!draws.at(-1).target&&pending.size===0,'ended stops RAF at480 with context status');
const old=fs.readFileSync(__dirname+'/ui_timing_previous/player.js','utf8');const now=fs.readFileSync(__dirname+'/player.js','utf8');ok(old.slice(0,old.indexOf('const api='))===now.slice(0,now.indexOf('function timingInfo')),'state schema version answers counts and lock core byteunchanged');
ok(now.includes("storageKey=VERSION+':'+materialId")&&old.includes("storageKey=VERSION+':'+materialId")&&!now.includes('removeItem(')&&!now.includes('localStorage.clear'),'same localStorage key no reset or clear');
console.log(JSON.stringify({status:'passed',count:checks.length,checks,actual_browser_playback_tested:false,human_responses_generated:0}));
