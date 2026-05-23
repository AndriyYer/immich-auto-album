#!/usr/bin/env python3
"""One-shot backlog runner: classify an entire Immich library and add matches
to a target album.

Intended to run on a fast machine (e.g. GPU box) before deploying the polling
sidecar to a slower host. For each asset:

- IMAGE: download the preview thumbnail and classify with YOLO11m.
- VIDEO: skip if duration > --max-video-seconds; otherwise fetch the
  transcoded MP4 (falling back to the original), sparse-sample frames, and
  early-terminate as soon as any frame scores cat >= --early-terminate-conf.

Match rule: cat_conf >= --cat-threshold OR dog_conf >= --dog-threshold. The
dog co-trigger catches photos YOLO confidently mislabels as dog.

Writes a CSV audit trail to --logs-dir before touching the album, then PUTs
matched asset IDs into the target album in chunks of 500. Run
`cache_from_csv.py` against the same CSV to produce a sidecar-compatible
cache JSON so the sidecar doesn't redo the work.
"""

import argparse
import csv
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

# Allow `python classifier/backlog.py ...` as well as `python -m classifier.backlog ...`
# by putting the repo root on sys.path when invoked as a script.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# `classifier.runtime` runs the Windows GPU DLL setup at import time, so it
# must be imported before `onnxruntime`.
from classifier.runtime import (
    CLASS_TO_ID,
    COCO_CLASSES,
    load_session,
    parse_duration,
    score_per_class,
)

import cv2
import numpy as np
import requests

CAT_ID = CLASS_TO_ID['cat']
DOG_ID = CLASS_TO_ID['dog']
PAGE_SIZE = 1000
ALBUM_CHUNK = 500


def score_frame(session, input_name, img_bgr):
    """Return (cat_conf, dog_conf, top_other_class_idx, top_other_conf)."""
    per_class = score_per_class(session, input_name, img_bgr)
    cat = per_class[CAT_ID]
    dog = per_class[DOG_ID]
    masked = per_class.copy()
    masked[CAT_ID] = -1.0
    masked[DOG_ID] = -1.0
    top_idx = int(masked.argmax())
    return cat, dog, top_idx, float(masked[top_idx])


class Immich:
    def __init__(self, base_url, api_key):
        self.base = base_url.rstrip('/')
        self.s = requests.Session()
        self.s.headers['x-api-key'] = api_key

    def iter_assets(self):
        page = 1
        while True:
            r = self.s.post(
                f'{self.base}/api/search/metadata',
                json={'page': page, 'size': PAGE_SIZE},
                timeout=30,
            )
            r.raise_for_status()
            assets = r.json().get('assets', {})
            items = assets.get('items', [])
            if not items:
                return
            yield from items
            nxt = assets.get('nextPage')
            if nxt is None:
                return
            page = int(nxt)

    def fetch_thumbnail(self, asset_id):
        r = self.s.get(
            f'{self.base}/api/assets/{asset_id}/thumbnail',
            params={'size': 'preview'},
            timeout=30,
        )
        r.raise_for_status()
        return r.content

    def stream_video(self, asset_id, out_path):
        """Try transcoded first, fall back to original. Returns label used."""
        attempts = [
            ('transcoded', f'{self.base}/api/assets/{asset_id}/video/playback'),
            ('original',   f'{self.base}/api/assets/{asset_id}/original'),
        ]
        last_err = None
        for label, url in attempts:
            try:
                with self.s.get(url, stream=True, timeout=180) as r:
                    if r.status_code != 200:
                        last_err = f'{label} HTTP {r.status_code}'
                        continue
                    with open(out_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=1024 * 256):
                            f.write(chunk)
                    return label
            except Exception as e:
                last_err = f'{label} {e!r}'
        raise RuntimeError(f'video fetch failed: {last_err}')

    def add_to_album(self, album_id, asset_ids):
        r = self.s.put(
            f'{self.base}/api/albums/{album_id}/assets',
            json={'ids': list(asset_ids)},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()


def classify_image(session, input_name, immich, asset_id):
    blob = immich.fetch_thumbnail(asset_id)
    arr = np.frombuffer(blob, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError('thumbnail decode failed')
    cat, dog, top_idx, top_conf = score_frame(session, input_name, img)
    return cat, dog, top_idx, top_conf, 1, 'thumbnail'


def classify_video(session, input_name, immich, asset_id, max_frames, sample_period_s, early_term):
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.mp4')
    os.close(tmp_fd)
    try:
        source = immich.stream_video(asset_id, tmp_path)
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise RuntimeError('video open failed')
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(fps * sample_period_s)))
        best_cat, best_dog, best_top, best_top_conf = 0.0, 0.0, 0, 0.0
        scored = 0
        idx = 0
        while scored < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                c, d, t, tc = score_frame(session, input_name, frame)
                scored += 1
                if c > best_cat:
                    best_cat = c
                if d > best_dog:
                    best_dog = d
                if tc > best_top_conf:
                    best_top, best_top_conf = t, tc
                if best_cat >= early_term:
                    break
            idx += 1
        cap.release()
        return best_cat, best_dog, best_top, best_top_conf, scored, source
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--server', required=True, help='Immich server URL, e.g. http://immich:2283')
    ap.add_argument('--api-key-env', required=True, help='Env var name holding the Immich API key')
    ap.add_argument('--album-id', required=True, help='Target album UUID')
    ap.add_argument('--label', default='run', help='Used in the output CSV filename to distinguish runs')
    ap.add_argument('--model', default=str(Path(__file__).parent / 'models' / 'yolo11m.onnx'))
    ap.add_argument('--cat-threshold', type=float, default=0.30)
    ap.add_argument('--dog-threshold', type=float, default=0.70)
    ap.add_argument('--early-terminate-conf', type=float, default=0.70)
    ap.add_argument('--max-video-seconds', type=float, default=300.0)
    ap.add_argument('--video-max-frames', type=int, default=20)
    ap.add_argument('--video-sample-period', type=float, default=2.0)
    ap.add_argument('--logs-dir', default=str(Path(__file__).parent / 'logs'))
    ap.add_argument('--limit', type=int, default=0, help='process at most N assets (0 = all)')
    ap.add_argument('--skip-album-write', action='store_true', help='write CSV but do not touch album')
    args = ap.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        sys.exit(f'missing env var {args.api_key_env}')

    immich = Immich(args.server, api_key)

    session = load_session(args.model, prefer_gpu=True)
    input_name = session.get_inputs()[0].name
    print(f'label={args.label}  album={args.album_id}', flush=True)
    print(f'model={args.model}', flush=True)
    print(f'providers={session.get_providers()}', flush=True)
    print(f'thresholds: cat>={args.cat_threshold}  dog>={args.dog_threshold}  early_term>={args.early_terminate_conf}', flush=True)
    print(f'video: max_seconds={args.max_video_seconds}  max_frames={args.video_max_frames}  period={args.video_sample_period}s', flush=True)

    Path(args.logs_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    csv_path = Path(args.logs_dir) / f'run-{args.label}-{ts}.csv'

    rows = []
    matched_ids = []
    counts = {
        'images': 0, 'videos': 0,
        'skipped_trashed': 0, 'skipped_archived': 0,
        'skipped_video_too_long': 0, 'skipped_unknown_kind': 0,
        'errors': 0,
        'matched_cat': 0, 'matched_dog_only': 0,
    }
    t0 = time.time()
    seen = 0
    for asset in immich.iter_assets():
        seen += 1
        if args.limit and seen > args.limit:
            break

        asset_id = asset['id']
        kind = asset.get('type', '')
        fname = asset.get('originalFileName', '')
        duration_s = parse_duration(asset.get('duration'))

        row = {
            'asset_id': asset_id, 'kind': kind, 'filename': fname,
            'duration_s': f'{duration_s:.2f}' if kind == 'VIDEO' else '',
            'cat_conf': '', 'dog_conf': '',
            'top_other_class': '', 'top_other_conf': '',
            'frames_scored': 0, 'video_source': '',
            'matched': '', 'error': '',
        }

        if asset.get('isTrashed'):
            row['error'] = 'skip: trashed'
            counts['skipped_trashed'] += 1
            rows.append(row)
            continue
        if asset.get('isArchived'):
            row['error'] = 'skip: archived'
            counts['skipped_archived'] += 1
            rows.append(row)
            continue

        try:
            if kind == 'IMAGE':
                cat, dog, top_idx, top_conf, frames, src = classify_image(session, input_name, immich, asset_id)
                counts['images'] += 1
            elif kind == 'VIDEO':
                if duration_s > args.max_video_seconds:
                    row['error'] = f'skip: duration {duration_s:.0f}s > {args.max_video_seconds:.0f}s'
                    counts['skipped_video_too_long'] += 1
                    rows.append(row)
                    _maybe_progress(seen, t0, counts, matched_ids)
                    continue
                cat, dog, top_idx, top_conf, frames, src = classify_video(
                    session, input_name, immich, asset_id,
                    args.video_max_frames, args.video_sample_period, args.early_terminate_conf,
                )
                counts['videos'] += 1
            else:
                row['error'] = f'skip: unknown kind {kind!r}'
                counts['skipped_unknown_kind'] += 1
                rows.append(row)
                continue

            row['cat_conf'] = f'{cat:.4f}'
            row['dog_conf'] = f'{dog:.4f}'
            row['top_other_class'] = COCO_CLASSES[top_idx] if 0 <= top_idx < len(COCO_CLASSES) else ''
            row['top_other_conf'] = f'{top_conf:.4f}'
            row['frames_scored'] = frames
            row['video_source'] = src if kind == 'VIDEO' else ''

            cat_hit = cat >= args.cat_threshold
            dog_hit = dog >= args.dog_threshold
            if cat_hit or dog_hit:
                if cat_hit and dog_hit:
                    row['matched'] = 'cat+dog'
                    counts['matched_cat'] += 1
                elif cat_hit:
                    row['matched'] = 'cat'
                    counts['matched_cat'] += 1
                else:
                    row['matched'] = 'dog_only'
                    counts['matched_dog_only'] += 1
                matched_ids.append(asset_id)
        except Exception as e:
            row['error'] = repr(e)
            counts['errors'] += 1

        rows.append(row)
        _maybe_progress(seen, t0, counts, matched_ids)

    elapsed = time.time() - t0
    print(f'\nclassified {seen} assets in {elapsed:.1f}s ({seen / max(elapsed, 0.001):.1f}/s)', flush=True)

    with csv_path.open('w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'asset_id', 'kind', 'filename', 'duration_s',
            'cat_conf', 'dog_conf', 'top_other_class', 'top_other_conf',
            'frames_scored', 'video_source', 'matched', 'error',
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f'csv: {csv_path}', flush=True)

    matched_ids = list(dict.fromkeys(matched_ids))
    print(f'\nmatched {len(matched_ids)} assets ({counts["matched_cat"]} cat, {counts["matched_dog_only"]} dog-only)', flush=True)

    added = 0
    duplicates = 0
    add_errors = 0
    if args.skip_album_write:
        print('--skip-album-write set, not adding to album', flush=True)
    elif not matched_ids:
        print('nothing matched, nothing to add', flush=True)
    else:
        for i in range(0, len(matched_ids), ALBUM_CHUNK):
            chunk = matched_ids[i:i + ALBUM_CHUNK]
            try:
                result = immich.add_to_album(args.album_id, chunk)
                ok = sum(1 for r in result if r.get('success'))
                dup = sum(1 for r in result if not r.get('success') and r.get('error') == 'duplicate')
                err = sum(1 for r in result if not r.get('success') and r.get('error') != 'duplicate')
                added += ok
                duplicates += dup
                add_errors += err
                print(f'  chunk {i // ALBUM_CHUNK + 1}: +{ok}  dup={dup}  err={err}', flush=True)
            except Exception as e:
                add_errors += len(chunk)
                print(f'  chunk {i // ALBUM_CHUNK + 1}: REQUEST FAILED {e!r}', flush=True)

    print('\n--- summary ---')
    for k, v in counts.items():
        print(f'  {k}: {v}')
    print(f'  total_seen: {seen}')
    print(f'  matched_total: {len(matched_ids)}')
    print(f'  album_added: {added}')
    print(f'  album_duplicates: {duplicates}')
    print(f'  album_errors: {add_errors}')
    print(f'  csv: {csv_path}')


def _maybe_progress(seen, t0, counts, matched_ids):
    if seen % 200 != 0:
        return
    elapsed = time.time() - t0
    rate = seen / max(elapsed, 0.001)
    print(
        f'[{seen}] {rate:.1f}/s  '
        f'img={counts["images"]} vid={counts["videos"]} '
        f'skip_vid_long={counts["skipped_video_too_long"]} '
        f'matched={len(matched_ids)} (cat={counts["matched_cat"]} dog_only={counts["matched_dog_only"]}) '
        f'errs={counts["errors"]}',
        flush=True,
    )


if __name__ == '__main__':
    main()
