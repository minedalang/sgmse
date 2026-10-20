const player = document.getElementById('player');
const current = document.getElementById('current');
let generation=0, request=null, objectURL=null;
function stopPlayback(){
 generation++;
 if(request) request.abort(); request=null;
 player.pause();player.removeAttribute('src');player.load();
 if(objectURL) URL.revokeObjectURL(objectURL);objectURL=null;
}
async function selectClip(clip){
 stopPlayback();const ticket=generation,controller=new AbortController();request=controller;
 current.textContent='正在载入：'+clip.name;
 try{
  const response=await fetch(clip.file,{signal:controller.signal});
  if(!response.ok) throw new Error('HTTP '+response.status);
  const blob=await response.blob();
  if(ticket!==generation||controller.signal.aborted)return;
  objectURL=URL.createObjectURL(blob);player.src=objectURL;
  current.textContent='当前片段：'+clip.name;await player.play();
 }catch(error){
  if(ticket!==generation||controller.signal.aborted)return;
  current.textContent='当前片段：'+clip.name+'（未自动播放，可按播放器播放；若未载入请重试）';
 }finally{if(ticket===generation)request=null;}
}
document.getElementById('stop').onclick=()=>{stopPlayback();current.textContent='已停止，请选择片段';};
document.querySelectorAll('button[data-file]').forEach(button=>{
 button.onclick=()=>selectClip({file:button.dataset.file,name:button.dataset.name});
});
window.addEventListener('pagehide',stopPlayback);
