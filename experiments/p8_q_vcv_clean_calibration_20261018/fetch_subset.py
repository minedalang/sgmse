#!/usr/bin/env python3
"""Fetch a frozen, metadata-selected clean subset; never run a model or save a ZIP.

Run once with --output <new-directory>. Unsupported ranges are a hard failure.
Only requests is nonstdlib. Defaults use the already audited local source files.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import time
import wave
import zipfile
import zlib

import requests

NETWORK_CAP = 8_000_000
DERIVED_CAP = 32_000_000
NAME = re.compile(r"(?P<sex>[fm])(?P<speaker_number>\d+)(?P<vowel1>[aiu])(?P<consonant>[a-z]+)(?P<vowel2>[aiu])(?P<stress>[12])\.wav", re.I)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    current_size = sum(p.stat().st_size for p in path.parent.rglob("*") if p.is_file())
    if current_size + len(encoded.encode("utf-8")) > DERIVED_CAP:
        raise RuntimeError("32 MB derived output budget would be exceeded")
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)


class RangeClient:
    def __init__(self):
        self.received = 0
        self.proxy_used = False
        self.attempts = []
        self.session = requests.Session()
        self.session.trust_env = False

    def get(self, url, start, stop, total):
        """Return exactly [start, stop); a single global proxy retry is allowed."""
        if not 0 <= start < stop <= total:
            raise ValueError("Invalid range")
        if stop - start > NETWORK_CAP - self.received:
            raise RuntimeError("8 MB received-body budget would be exceeded")
        try:
            return self._get(url, start, stop, total, False)
        except (requests.RequestException, ValueError) as exc:
            if self.proxy_used:
                raise
            self.proxy_used = True
            self.attempts[-1]["direct_failure"] = str(exc)
            return self._get(url, start, stop, total, True)

    def _get(self, url, start, stop, total, proxy):
        item = {"url": url, "start": start, "stop_exclusive": stop,
                "mode": "proxy" if proxy else "direct", "bytes_received": 0}
        self.attempts.append(item)
        proxies = {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"} if proxy else {}
        try:
            # No automatic redirect can create a second proxy fallback request.
            with self.session.get(url, headers={"Range": f"bytes={start}-{stop-1}",
                                  "Accept-Encoding": "identity"}, stream=True,
                                  timeout=30, proxies=proxies, allow_redirects=False) as response:
                item.update(status=response.status_code, final_url=response.url,
                            content_range=response.headers.get("Content-Range"))
                if response.status_code != 206:
                    raise ValueError(f"Range unsupported: HTTP {response.status_code}; response body refused")
                expected = f"bytes {start}-{stop-1}/{total}"
                if response.headers.get("Content-Range") != expected:
                    raise ValueError(f"Content-Range mismatch: expected {expected!r}")
                if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                    raise ValueError("Encoded range response refused")
                declared = response.headers.get("Content-Length")
                if declared is not None and int(declared) != stop - start:
                    raise ValueError("Content-Length mismatch")
                chunks = []
                length = 0
                # raw reads prevent requests from transparently decoding the body.
                while length < stop - start:
                    remaining = NETWORK_CAP - self.received
                    if remaining <= 0:
                        raise RuntimeError("8 MB received-body cap reached")
                    chunk = response.raw.read(min(8192, stop - start - length, remaining))
                    if not chunk:
                        raise ValueError("Truncated range response")
                    self.received += len(chunk)
                    item["bytes_received"] += len(chunk)
                    length += len(chunk)
                    chunks.append(chunk)
                item["success"] = True
                return b"".join(chunks)
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
            raise


class RangeFile(io.RawIOBase):
    """The seekable read interface used by zipfile, with no archive materialization."""
    def __init__(self, client, url, size):
        super().__init__()
        self.client, self.url, self.size, self.position = client, url, size, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size if whence == 2 else -10**30)
        if position < 0:
            raise ValueError("Invalid seek")
        self.position = position
        return position

    def read(self, size=-1):
        stop = self.size if size is None or size < 0 else min(self.size, self.position + size)
        if stop <= self.position:
            return b""
        data = self.client.get(self.url, self.position, stop, self.size)
        self.position = stop
        return data


def fields(name):
    match = NAME.fullmatch(PurePosixPath(name).name)
    if not match:
        return None
    result = match.groupdict()
    result["sex"] = result["sex"].lower()
    result["speaker_number"] = int(result["speaker_number"])
    result["speaker"] = f"{result['sex']}{result['speaker_number']}"
    result["context"] = result["vowel1"] + "C" + result["vowel2"]
    return result


def manual_segments(path):
    targets, current = {}, None
    for line in path.read_text().splitlines():
        if line.startswith('"'):
            current = PurePosixPath(line.strip('"')).stem.lower() + ".wav"
            targets[current] = []
        elif current and re.match(r"^\d+\s+\d+\s+", line):
            start, end, label = line.split()[:3]
            targets[current].append({"start_htk": int(start), "end_htk": int(end),
                                     "start_seconds": int(start) / 1e7,
                                     "end_seconds": int(end) / 1e7, "label": label})
    return {name: segments for name, segments in targets.items()
            if any(segment["label"] in ("k", "g", "t", "d") for segment in segments)}


def select(inventory, manual):
    indexed = {}
    for row in inventory:
        parsed = row.get("nominal")
        if parsed:
            key = row["basename"].lower()
            if key in indexed:
                raise ValueError(f"Ambiguous duplicate audio basename: {key}")
            indexed[key] = row
    speakers = sorted({r["nominal"]["speaker"] for r in indexed.values()},
                      key=lambda s: (int(s[1:]), s[0]))
    eligible = {}
    for speaker in speakers:
        for vowel in "aiu":
            for stress in "12":
                names = [f"{speaker}{vowel}{c}{vowel}{stress}.wav" for c in "kgtd"]
                if all(name in indexed for name in names):
                    eligible[speaker] = names
                    break
            if speaker in eligible:
                break
    queues = {sex: [s for s in speakers if s[0] == sex and s in eligible] for sex in "fm"}
    chosen = []
    while len(chosen) < 4 and any(queues.values()):
        for sex in "fm":
            if queues[sex] and len(chosen) < 4:
                chosen.append(queues[sex].pop(0))
    if len(chosen) != 4:
        raise ValueError(f"Need four eligible train speakers; found {list(eligible)}")
    available_sexes = {s[0] for s in speakers}
    if len(available_sexes) >= 2 and len({s[0] for s in chosen}) < 2:
        raise ValueError("Both sexes occur in archive but matched selection cannot include both")
    primary = [indexed[name] for speaker in chosen for name in eligible[speaker]]
    primary_names = {r["basename"].lower() for r in primary}
    supplementary = [indexed[name] for name in sorted(manual)
                     if name in indexed and name not in primary_names][:13]
    rows = []
    for row in primary + supplementary:
        row = dict(row)
        name = row["basename"].lower()
        row["role"] = "primary" if name in primary_names else "supplementary_manual"
        row["manual_boundaries"] = manual.get(name)
        rows.append(row)
    pairs = [{"pair_id": speaker + "_" + contrast.replace("/", ""),
              "speaker": speaker, "contrast": contrast,
              "members": eligible[speaker][offset:offset + 2],
              "context": indexed[eligible[speaker][offset]]["nominal"]["context"],
              "stress": indexed[eligible[speaker][offset]]["nominal"]["stress"]}
             for speaker in chosen for contrast, offset in (("k/g", 0), ("t/d", 2))]
    return {"speakers": chosen, "pairs": pairs, "files": rows,
            "manual_targets_in_archive": sorted(set(manual) & set(indexed)),
            "identity_basis": "Corpus filename nominal labels, independent of teacher/model results",
            "manual_boundary_caveat": "Hand segmentation is supplementary; it does not establish perfect consonant identity",
            "frozen_before_model_results": True}


def run(args):
    # mkdir(exist_ok=False) is the run-level write-once guard, including failures.
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    client = RangeClient()
    started = time.time()
    status = {"phase": "selection_only", "training_or_inference": False,
              "status": "started", "source_data_modified": False}
    try:
        record_bytes = args.metadata.read_bytes()
        record = json.loads(record_bytes)
        archive = next(f for f in record["files"] if f["key"] == "train.zip")
        if archive["size"] != 169755879:
            raise ValueError("Audited train.zip size changed")
        url = args.url or archive["links"]["self"]
        provenance = {"record_id": record["id"], "doi": record["doi"],
                      "license": record["metadata"]["license"], "source_url": url,
                      "archive_metadata": archive, "record_sha256": sha(record_bytes),
                      "manual_mlf_sha256": sha(args.manual_mlf.read_bytes()),
                      "archive_checksum_verified": False,
                      "checksum_note": "Full archive MD5 is metadata only; each extracted WAV CRC32 is verified"}
        write_json(output / "provenance.json", provenance)
        with zipfile.ZipFile(RangeFile(client, url, archive["size"])) as source:
            inventory = []
            for member in source.infolist():
                if member.is_dir():
                    continue
                path = PurePosixPath(member.filename)
                if path.is_absolute() or ".." in path.parts or "\\" in member.filename:
                    raise ValueError(f"Unsafe archive member path: {member.filename!r}")
                inventory.append({"member": member.filename, "basename": path.name,
                                  "size": member.file_size, "compressed_size": member.compress_size,
                                  "crc32": f"{member.CRC:08x}", "nominal": fields(member.filename)})
            write_json(output / "central_directory_inventory.json", inventory)
            frozen = select(inventory, manual_segments(args.manual_mlf))
            for row in frozen["files"]:
                nominal = row["nominal"]
                segments = row["manual_boundaries"] or []
                target = next((s for s in segments if s["label"] == nominal["consonant"]), None)
                row.update(sample_id=Path(row["basename"]).stem,
                           wav_path=str(output / "raw_audio" / row["basename"]),
                           split="corpus_train",
                           speaker=nominal["speaker"], gender=nominal["sex"],
                           vowel_left=nominal["vowel1"], vowel_right=nominal["vowel2"],
                           stress=nominal["stress"], consonant=nominal["consonant"],
                           primary_pair_ids=[p["pair_id"] for p in frozen["pairs"]
                                             if row["basename"].lower() in p["members"]],
                           is_primary=row["role"] == "primary",
                           manual_start_s=target["start_seconds"] if target else None,
                           manual_end_s=target["end_seconds"] if target else None,
                           boundary_source="manual_mlf" if target else "none")
            if len(frozen["files"]) > 32:
                raise ValueError("32 WAV cap exceeded")
            frozen["provenance"] = provenance
            # Freeze on disk before reading any selected waveform, and before model results.
            write_json(output / "selection_plan_frozen.json", frozen)
            columns = ["sample_id", "wav_path", "split", "speaker", "gender", "vowel_left", "vowel_right",
                       "stress", "consonant", "primary_pair_ids", "is_primary", "manual_start_s",
                       "manual_end_s", "boundary_source", "sha256"]
            derived_size = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
            if derived_size + sum(r["size"] for r in frozen["files"]) + 250_000 > DERIVED_CAP:
                raise ValueError("32 MB derived output budget would be exceeded")
            audio_dir = output / "raw_audio"
            audio_dir.mkdir()
            extracted = []
            for row in frozen["files"]:
                if row["size"] > 2_000_000:
                    raise ValueError("Unexpected individual WAV size")
                payload = source.read(row["member"])  # zipfile also verifies CRC32.
                crc = f"{zlib.crc32(payload) & 0xffffffff:08x}"
                if len(payload) != row["size"] or crc != row["crc32"]:
                    raise ValueError("WAV size/CRC mismatch")
                with wave.open(io.BytesIO(payload), "rb") as wav:
                    wav_metadata = {"sample_rate": wav.getframerate(), "channels": wav.getnchannels(),
                                    "sample_width": wav.getsampwidth(), "frames": wav.getnframes()}
                    if wav.getframerate() != 25000:
                        raise ValueError("Expected original 25 kHz WAV; no resampling is performed")
                with (audio_dir / row["basename"]).open("xb") as stream:
                    stream.write(payload)
                row["sha256"] = sha(payload)
                extracted.append({"member": row["member"], "output": "raw_audio/" + row["basename"],
                                  "sha256": sha(payload), "crc32": crc, "crc32_verified": True,
                                  "bytes": len(payload), **wav_metadata})
            write_json(output / "audio_integrity.json", extracted)
            write_json(output / "selected_manifest_frozen.json", frozen)
            with (output / "selected_manifest_frozen.csv").open("x", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                for row in frozen["files"]:
                    flat = {key: row[key] for key in columns}
                    flat["primary_pair_ids"] = "|".join(flat["primary_pair_ids"])
                    writer.writerow(flat)
            manifest_sha = sha((output / "selected_manifest_frozen.json").read_bytes())
        status.update(status="completed_selection_only", selected_wavs=len(extracted),
                      primary_pairs=8, manifest_sha256=manifest_sha)
    except Exception as exc:
        status.update(status="failed", failure_type=type(exc).__name__, failure_reason=str(exc))
        raise
    finally:
        status.update(network_body_bytes=client.received, network_cap_bytes=NETWORK_CAP,
                      proxy_fallback_used=client.proxy_used, attempts=client.attempts,
                      elapsed_seconds=round(time.time() - started, 3))
        write_json(output / "metrics.json", status)
        with (output / "experiment_report.md").open("x") as stream:
            stream.write("# VCV clean subset retrieval\n\n" + json.dumps(status, indent=2) + "\n\n"
                         "Selection and unchanged raw WAV retrieval only. No training or enhancement inference. "
                         "Corpus labels are nominal identities; manual segments provide supplementary boundaries.\n")
        if sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) > DERIVED_CAP:
            raise RuntimeError("Derived output cap exceeded")


def main():
    here = Path(__file__).resolve().parent
    audit = here.parent / "p8_q_external_corpus_audit_20261018"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=here / "retrieval")
    parser.add_argument("--metadata", type=Path, default=audit / "record.json")
    parser.add_argument("--manual-mlf", type=Path, default=audit / "handsegm.91.mlf.txt")
    parser.add_argument("--url", choices=("https://zenodo.org/api/records/3538223/files/train.zip/content",
                                         "https://zenodo.org/records/3538223/files/train.zip?download=1"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
