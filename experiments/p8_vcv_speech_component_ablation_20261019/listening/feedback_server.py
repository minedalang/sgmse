"""Static experiment server with one durable, scoped feedback API. Standard library only."""
import argparse
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
from urllib.parse import parse_qs, unquote, urlsplit
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
API = '/' + HERE.parent.name + '/listening/api/feedback'
CASE_IDS = {'original__N1_snr-6__m2ada1', 'fresh__N1_snr-6__f20aka1', 'original__N2_snr-6__m2aga1', 'original__N1_snr-6__f11ata1', 'fresh__N2_snr-6__f20aka1'}
VALUES = {'unrated', 'native', 'ablated', 'similar', 'uncertain'}
LOCK = threading.Lock()

def valid_id(value):
    try:
        return isinstance(value, str) and str(uuid.UUID(value)) == value
    except (ValueError, AttributeError):
        return False

def validated(data):
    if not isinstance(data, dict) or set(data) != {'session_id', 'response_id', 'choices', 'note'}:
        raise ValueError('invalid fields')
    if not all(valid_id(data[k]) for k in ('session_id', 'response_id')):
        raise ValueError('invalid ID')
    if not isinstance(data['choices'], dict) or not data['choices'] or not set(data['choices']).issubset(CASE_IDS):
        raise ValueError('one or more of the five fixed case IDs required')
    if not all(isinstance(v, str) and v in VALUES for v in data['choices'].values()):
        raise ValueError('invalid choice')
    if not isinstance(data['note'], str) or len(data['note']) > 2000:
        raise ValueError('invalid note')
    return data

class Handler(SimpleHTTPRequestHandler):
    def reply(self, code, data):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def records(self):
        if not self.server.record_file.exists():
            return []
        return [json.loads(line) for line in self.server.record_file.read_text().splitlines() if line.strip()]

    def forbidden_path(self):
        path = unquote(urlsplit(self.path).path)
        return '..' in path.split('/') or path.startswith('/' + HERE.parent.name + '/listening/responses')

    def do_GET(self):
        parsed = urlsplit(self.path)
        if parsed.path == API:
            query = parse_qs(parsed.query)
            if set(query) != {'session_id'} or len(query['session_id']) != 1 or not valid_id(query['session_id'][0]):
                return self.reply(400, {'error': 'valid session_id required'})
            with LOCK:
                rows = [r for r in self.records() if r['session_id'] == query['session_id'][0]]
            return self.reply(200, {'ok': True, 'record': rows[-1] if rows else None})
        if self.forbidden_path():
            return self.reply(403, {'error': 'forbidden path'})
        return super().do_GET()

    def do_HEAD(self):
        if self.forbidden_path():
            return self.reply(403, {'error': 'forbidden path'})
        return super().do_HEAD()

    def end_headers(self):
        if urlsplit(self.path).path.endswith(('index.html', 'feedback.js')):
            self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

    def do_POST(self):
        if urlsplit(self.path).path != API or urlsplit(self.path).query:
            return self.reply(404, {'error': 'unknown endpoint'})
        if self.headers.get('Origin') and self.headers['Origin'] != 'http://' + self.headers.get('Host', ''):
            return self.reply(403, {'error': 'same origin required'})
        if self.headers.get('Content-Type', '').split(';')[0].strip() != 'application/json':
            return self.reply(415, {'error': 'application/json required'})
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if size <= 0 or size > 16384:
                return self.reply(413, {'error': 'payload size limit: 16384 bytes'})
            data = validated(json.loads(self.rfile.read(size)))
        except (ValueError, TypeError, UnicodeError):
            return self.reply(400, {'error': 'invalid feedback payload'})
        try:
            with LOCK:
                for old in self.records():
                    if old['response_id'] == data['response_id']:
                        if any((old.get('submitted_choices', old['choices']) if k == 'choices' else old[k]) != data[k] for k in data):
                            return self.reply(409, {'error': 'response_id already used with different content'})
                        return self.reply(200, {'ok': True, 'duplicate': True, 'record': old})
                prior = [r for r in self.records() if r['session_id'] == data['session_id']]
                merged = dict(prior[-1]['choices']) if prior else {}
                merged.update(data['choices'])
                record = dict(data, choices=merged, submitted_choices=data['choices'], timestamp=datetime.now(timezone.utc).isoformat(), protocol='prompted/nonblind/singlelistener', test_record=self.server.test_mode)
                self.server.record_file.parent.mkdir(exist_ok=True)
                with self.server.record_file.open('a', encoding='utf-8') as f:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    f.flush()
                    os.fsync(f.fileno())
            return self.reply(201, {'ok': True, 'duplicate': False, 'record': record})
        except OSError:
            return self.reply(500, {'error': 'durable write failed; retry with same response_id'})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--test-mode', action='store_true')
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(Handler, directory=str(ROOT)))
    server.test_mode = args.test_mode
    server.record_file = HERE / 'responses' / ('test_records.jsonl' if args.test_mode else 'feedback.jsonl')
    server.serve_forever()

if __name__ == '__main__':
    main()
