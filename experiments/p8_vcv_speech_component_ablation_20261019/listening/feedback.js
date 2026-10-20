// Drafts survive reloads; only a matching durable server acknowledgment means saved.
(() => {
  const key = 'vcv-speech-ablation-feedback-v1';
  const api = 'api/feedback';
  const status = document.getElementById('feedback-status');
  const button = document.getElementById('save-feedback');
  const note = document.getElementById('feedback-note');
  const selects = Array.from(document.querySelectorAll('select[data-case-id]'));
  const uuid = () => crypto.randomUUID();
  const validValues = new Set(['unrated', 'native', 'ablated', 'similar', 'uncertain']);
  let storageOK = true, revision = 0, inFlight = false;
  let draft = null;
  try { draft = JSON.parse(localStorage.getItem(key)); } catch (_) { storageOK = false; }
  if (!draft || !/^[0-9a-f-]{36}$/.test(draft.session_id || '') || !draft.choices || !Object.values(draft.choices).every(v => validValues.has(v))) {
    draft = {session_id: uuid(), response_id: uuid(), choices: Object.fromEntries(selects.map(s => [s.getAttribute('data-case-id'), 'unrated'])), note: '', saved: false};
  }
  selects.forEach(s => { const id = s.getAttribute('data-case-id'); if (!(id in draft.choices)) draft.choices[id] = 'unrated'; });
  function show(text) { status.textContent = text + (storageOK ? '' : '（浏览器草稿存储不可用，请勿关闭页面）'); }
  function persist() { try { localStorage.setItem(key, JSON.stringify(draft)); } catch (_) { storageOK = false; } }
  function render() { selects.forEach(s => { s.value = draft.choices[s.getAttribute('data-case-id')]; }); note.value = draft.note || ''; }
  function changed() {
    revision++;
    draft.choices = Object.fromEntries(selects.map(s => [s.getAttribute('data-case-id'), s.value]));
    draft.note = note.value;
    draft.response_id = uuid(); draft.saved = false;
    persist(); show('未保存到服务器：草稿已保留，请按“保存选择”。');
  }
  async function request(url, options) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 8000);
    try {
      const response = await fetch(url, {...options, signal: controller.signal, cache: 'no-store'});
      const body = await response.json();
      if (!response.ok || !body.ok) throw new Error(body.error || 'HTTP ' + response.status);
      return body;
    } finally { clearTimeout(timer); }
  }
  function matches(record, payload) {
    return record && record.session_id === payload.session_id && record.response_id === payload.response_id && record.note === payload.note && Object.entries(payload.choices).filter(([,v]) => v !== 'unrated').every(([id,v]) => record.choices[id] === v);
  }
  selects.forEach(s => s.addEventListener('change', changed));
  note.addEventListener('input', changed);
  render(); persist();
  show(draft.saved ? '正在核对服务器保存记录…' : '本轮五例聊天反馈已记录；网页选项仅供日后自愿更新。');
  button.addEventListener('click', async () => {
    if (inFlight) return;
    inFlight = true; button.disabled = true;
    const version = ++revision;
    const payload = JSON.parse(JSON.stringify({session_id: draft.session_id, response_id: draft.response_id, choices: Object.fromEntries(Object.entries(draft.choices).filter(([,v]) => v !== 'unrated')), note: draft.note}));
    if (!Object.keys(payload.choices).length) { show('尚未选择新的判断；聊天确认的五例已单独记录，无需重复填写。'); inFlight = false; button.disabled = false; return; }
    show('保存中…尚未收到服务器确认。');
    try {
      const result = await request(api, {method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload)});
      if (!matches(result.record, payload)) throw new Error('服务器确认内容不匹配');
      if (version === revision) { draft.saved = true; persist(); show('已保存到服务器 ✓ ' + result.record.timestamp); }
      else show('上一版已保存；当前修改尚未保存，请再按“保存选择”。');
    } catch (_) { show('保存失败／未确认：选择与草稿仍保留，请按“保存选择”重试。'); }
    finally { inFlight = false; button.disabled = false; }
  });
  const restoreRevision = revision;
  const restoreID = draft.response_id;
  request(api + '?session_id=' + encodeURIComponent(draft.session_id)).then(result => {
    if (revision !== restoreRevision || inFlight || draft.response_id !== restoreID) return;
    if (matches(result.record, draft)) { draft.saved = true; persist(); show('已保存到服务器 ✓ ' + result.record.timestamp); }
    else if (result.record && draft.saved) {
      draft = {...result.record, choices: Object.fromEntries(selects.map(s => { const id=s.getAttribute('data-case-id'); return [id,result.record.choices[id] || 'unrated']; })), saved:true}; render(); persist(); show('已恢复服务器保存选择 ✓ ' + result.record.timestamp);
    } else { draft.saved = false; persist(); show('聊天反馈已单独记录；网页草稿保留，只有日后更新时才需保存。'); }
  }).catch(() => { if (revision === restoreRevision && !inFlight && draft.response_id === restoreID) show('暂时无法核对服务器：草稿保留，尚未确认保存，请重试保存。'); });
})();
