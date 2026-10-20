const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),crypto=require('node:crypto'),path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'feedback.js'),'utf8');
const ids=['original__N1_snr-6__m2ada1','fresh__N1_snr-6__f20aka1','original__N2_snr-6__m2aga1','original__N1_snr-6__f11ata1','fresh__N2_snr-6__f20aka1'];
const values=new Map(),requests=[];
const make=()=>({value:'',textContent:'',handlers:{},addEventListener(n,f){this.handlers[n]=f;}});
let selects,status,button,note;
function start(){
 selects=ids.map(id=>({...make(),getAttribute(){return id}}));status=make();button=make();note=make();
 const context={crypto,AbortController,setTimeout,clearTimeout,encodeURIComponent,localStorage:{getItem:k=>values.get(k)||null,setItem:(k,v)=>values.set(k,v)},document:{getElementById:id=>({'feedback-status':status,'save-feedback':button,'feedback-note':note}[id]),querySelectorAll:()=>selects},fetch:(url,options)=>new Promise((resolve,reject)=>requests.push({url,options,resolve,reject}))};
 vm.createContext(context);vm.runInContext(source,context);
}
const flush=()=>new Promise(resolve=>setImmediate(resolve));
const reply=(r,data,ok=true)=>r.resolve({ok,status:ok?200:500,json:async()=>data});
const draft=()=>JSON.parse([...values.values()][0]);
(async()=>{
 start();reply(requests.shift(),{ok:true,record:null});await flush();
 selects[0].value='native';selects[0].handlers.change();assert.equal(draft().choices[ids[0]],'native');
 assert(status.textContent.includes('未保存到服务器'));
 const attempt=button.handlers.click(), req=requests.shift(), payload=JSON.parse(req.options.body);
 assert(status.textContent.includes('保存中'));req.reject(new Error('offline'));await attempt;
 assert(status.textContent.includes('保存失败'));assert.equal(selects[0].value,'native');assert.equal(draft().response_id,payload.response_id);
 const retry=button.handlers.click(), r=requests.shift();assert.equal(JSON.parse(r.options.body).response_id,payload.response_id);
 reply(r,{ok:true,record:{...payload,timestamp:'test-time'}});await retry;assert(status.textContent.includes('已保存到服务器'));assert(draft().saved);
 start();reply(requests.shift(),{ok:true,record:{...payload,timestamp:'test-time'}});await flush();assert.equal(selects[0].value,'native');assert(status.textContent.includes('已保存到服务器'));
 selects[1].value='similar';selects[1].handlers.change();
 const saving=button.handlers.click(), pending=requests.shift(), old=JSON.parse(pending.options.body);
 selects[2].value='uncertain';selects[2].handlers.change();reply(pending,{ok:true,record:{...old,timestamp:'test-time'}});await saving;
 assert(status.textContent.includes('当前修改尚未保存'));assert.equal(draft().saved,false);assert.equal(selects[2].value,'uncertain');
 // A delayed restore response must not undo a completed save or replace a local draft.
 start();const delayed=requests.shift();assert.equal(selects[2].value,'uncertain');
 const finalSave=button.handlers.click(),finalRequest=requests.shift(),finalPayload=JSON.parse(finalRequest.options.body);
 reply(finalRequest,{ok:true,record:{...finalPayload,timestamp:'test-time'}});await finalSave;
 reply(delayed,{ok:true,record:null});await flush();assert(status.textContent.includes('已保存到服务器'));
 const historical={session_id:crypto.randomUUID(),response_id:crypto.randomUUID(),choices:{[ids[0]]:'similar',[ids[1]]:'native',[ids[2]]:'similar'},note:'historical test draft',saved:false};
 values.set('vcv-speech-ablation-feedback-v1',JSON.stringify(historical));
 start();reply(requests.shift(),{ok:true,record:null});await flush();
 assert.equal(selects[0].value,'similar');assert.equal(selects[1].value,'native');assert.equal(selects[2].value,'similar');assert.equal(selects[3].value,'unrated');assert.equal(draft().session_id,historical.session_id);
 selects[3].value='uncertain';selects[3].handlers.change();const supplement=button.handlers.click(),supplementReq=requests.shift(),supplementData=JSON.parse(supplementReq.options.body);assert(!('fresh__N2_snr-6__f20aka1' in supplementData.choices));reply(supplementReq,{ok:true,record:{...supplementData,timestamp:'test-time'}});await supplement;assert(status.textContent.includes('已保存到服务器'));
 const report={status:'passed',method:'Node VM minimal DOM, no browser automation',checks:['change persists local draft','explicit unsaved and saving status','network failure preserves choices and retry ID','retry acknowledgment required before saved','reload restores saved choices','editing during save remains unsaved','unsaved draft survives reload','late restore cannot undo a newer successful save','three-case draft migrates to five without choice/session loss','unrated added cases excluded from partial submission']};
 fs.writeFileSync(path.join(__dirname,'feedback_client_validation.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
})().catch(e=>{console.error(e);process.exitCode=1});
