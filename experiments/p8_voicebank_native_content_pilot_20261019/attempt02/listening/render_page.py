"""Render the completed listening page from immutable chat feedback; no audio writes."""
from pathlib import Path
import hashlib
import html
import json

HERE = Path(__file__).resolve().parent
FEEDBACK_SHA256 = '308eaca771c4a4b258a428cdb27c3f570580a42a89a5c1ec919724e52e700edc'

def render():
    raw = (HERE/'user_feedback_20261019.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FEEDBACK_SHA256
    feedback = json.loads(raw)
    assert feedback['source'] == 'user_chat'
    cases = {case['label']: case for case in feedback['cases']}
    page = (HERE/'index.template.html').read_text()
    for label in ['A', 'B']:
        token = '__FEEDBACK_' + label + '__'
        assert page.count(token) == 1
        page = page.replace(token, html.escape(cases[label]['comment_verbatim']))
    assert page.count('<audio ') == 1 and page.count('data-file=') == 8
    assert '<form' not in page and 'localStorage' not in page
    (HERE/'index.html').write_text(page)

if __name__ == '__main__':
    render()
