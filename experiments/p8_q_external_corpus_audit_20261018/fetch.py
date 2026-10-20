"""Bounded metadata fetch; no audio, one optional proxy fallback in total."""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
CAP = 2000000
used = 0
fallback_used = False
attempts = []
urls = [
 ('record.json', 'https://zenodo.org/api/records/3538223'),
 ('readme.pdf', 'https://zenodo.org/records/3538223/files/readme.pdf?download=1'),
 ('handsegm.91.mlf.txt', 'https://zenodo.org/records/3538223/files/handsegm.91.mlf.txt?download=1'),
]
if (ROOT / 'fetch_metrics.json').exists():
 raise FileExistsError('Existing fetch; do not repeat')
for name, url in urls:
 for mode in ('direct', 'proxy'):
  if mode == 'proxy':
   if fallback_used:
    break
   fallback_used = True
  if used >= CAP:
   break
  item = {'url': url, 'filename': name, 'mode': mode, 'status': None,
          'final_url': None, 'bytes_received': 0, 'success': False}
  started = time.monotonic()
  session = requests.Session()
  session.trust_env = False
  if mode == 'proxy':
   session.proxies = {'http': 'http://127.0.0.1:7890', 'https': 'http://127.0.0.1:7890'}
  try:
   with session.get(url, timeout=30, stream=True, headers={'Accept-Encoding': 'identity'}, allow_redirects=True) as response:
    item.update(status=response.status_code, final_url=response.url,
                content_type=response.headers.get('Content-Type'),
                content_length=response.headers.get('Content-Length'))
    # A failed HTTP response body is not needed and is not downloaded.
    response.raise_for_status()
    content_length = response.headers.get('Content-Length')
    directory_bytes = sum(p.stat().st_size for p in ROOT.iterdir() if p.is_file())
    remaining = min(CAP-used, CAP-directory_bytes-50000)
    if content_length and int(content_length) > remaining:
     raise ValueError('Declared content length exceeds remaining cap')
    data = bytearray()
    while len(data) < remaining:
     if time.monotonic()-started >= 30:
      raise TimeoutError('30 second request wall time limit reached')
     chunk = response.raw.read(min(16384, remaining-len(data)), decode_content=False)
     if not chunk:
      break
     data.extend(chunk)
     used += len(chunk)
     item['bytes_received'] += len(chunk)
    else:
     raise ValueError('Body reached cap; refusing truncated output')
    if content_length and len(data) != int(content_length):
     raise ValueError('Incomplete content-length body')
    (ROOT / name).write_bytes(data)
    item.update(success=True, saved_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
  except Exception as error:
   item['error'] = f'{type(error).__name__}: {error}'
  finally:
   item['elapsed_seconds'] = time.monotonic()-started
   attempts.append(item)
   session.close()
  if item['success']:
   break
result = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'cap_bytes': CAP,
          'network_body_bytes': used, 'proxy_fallback_used': fallback_used, 'attempts': attempts}
(ROOT / 'fetch_metrics.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
