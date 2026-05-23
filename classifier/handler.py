"""image_classifier source handler for the auto-album sidecar.

Polling cycle:
1. Page through every asset in the library.
2. If we have cached confidences for every class the rule currently thresholds,
   re-evaluate the match against current thresholds without re-running YOLO.
3. Otherwise (new asset, or rule added a new class to threshold), fetch a
   thumbnail (image) or sampled transcoded video frames, run YOLO, cache the
   per-class confs.
4. Return the set of asset IDs whose confs satisfy any rule threshold.

The cache stores raw confidences, not a match decision, so threshold tuning
doesn't invalidate cached work. Cache lives at
/app/data/classifier_cache/<rule_name>.json. To force a full reclassification,
delete that file.
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import requests

from classifier.runtime import (
    CLASS_TO_ID,
    load_session,
    parse_duration,
    score_per_class,
)

CACHE_DIR = Path(os.environ.get("CLASSIFIER_CACHE_DIR", "/app/data/classifier_cache"))
DEFAULT_MODEL_PATH = os.environ.get("CLASSIFIER_MODEL_PATH", "/app/models/yolo11m.onnx")
DEFAULT_VIDEO_OPTS = {
    "max_duration_seconds": 300.0,
    "max_frames": 20,
    "sample_period_seconds": 2.0,
    "early_terminate_conf": 0.70,
}
SEARCH_PAGE_SIZE = 250
# Flush cache to disk after this many newly-classified or evicted entries.
# Bounds work lost to a mid-cycle crash on long first scans (Celeron-class
# CPU + 10K-asset library is ~5 hours).
SAVE_EVERY = 200


def _cache_path(rule_name):
    return CACHE_DIR / f"{rule_name}.json"


def load_cache(rule_name, expected_model=None, logger=None):
    """Read the cache JSON for `rule_name`. If `expected_model` is given and
    the cache was built with a different model, discard it (the confidences
    would be from a different network and not comparable)."""
    p = _cache_path(rule_name)
    if not p.exists():
        return {}
    try:
        with p.open() as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    cached_model = data.get("model")
    if expected_model and cached_model and cached_model != expected_model:
        if logger:
            logger(
                f"rule {rule_name}: cache was built with model {cached_model!r} "
                f"but current model is {expected_model!r}; discarding to avoid "
                f"stale confidences",
                level="warn",
            )
        return {}
    return data.get("entries", {})


def save_cache(rule_name, entries, model_path):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = _cache_path(rule_name)
    tmp = p.with_suffix(".json.tmp")
    payload = {
        "version": 1,
        "model": Path(model_path).stem,
        "entries": entries,
    }
    with tmp.open("w") as f:
        json.dump(payload, f, separators=(",", ":"))
    tmp.replace(p)


def _now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fetch_thumbnail(client, asset_id):
    r = client.session.get(
        f"{client.server_url}/api/assets/{asset_id}/thumbnail",
        params={"size": "preview"},
        timeout=30,
    )
    r.raise_for_status()
    return r.content


def _stream_video(client, asset_id, out_path):
    """Try transcoded MP4 first, fall back to the original. Returns label used."""
    attempts = [
        ("transcoded", f"{client.server_url}/api/assets/{asset_id}/video/playback"),
        ("original",   f"{client.server_url}/api/assets/{asset_id}/original"),
    ]
    last_err = None
    for label, url in attempts:
        try:
            with client.session.get(url, stream=True, timeout=180) as r:
                if r.status_code != 200:
                    last_err = f"{label} HTTP {r.status_code}"
                    continue
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=256 * 1024):
                        f.write(chunk)
                return label
        except requests.RequestException as e:
            last_err = f"{label} {e!r}"
    raise RuntimeError(f"video fetch failed: {last_err}")


def _classify_image(session, input_name, client, asset_id, classes):
    blob = _fetch_thumbnail(client, asset_id)
    arr = np.frombuffer(blob, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("thumbnail decode failed")
    per_class = score_per_class(session, input_name, img)
    confs = {name: float(per_class[CLASS_TO_ID[name]]) for name in classes}
    return confs, 1


def _classify_video(session, input_name, client, asset_id, classes, video_opts):
    early_term = video_opts["early_terminate_conf"]
    max_frames = video_opts["max_frames"]
    period_s = video_opts["sample_period_seconds"]

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(tmp_fd)
    try:
        _stream_video(client, asset_id, tmp_path)
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise RuntimeError("video open failed")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(fps * period_s)))
        best_confs = {name: 0.0 for name in classes}
        scored = 0
        idx = 0
        while scored < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                per_class = score_per_class(session, input_name, frame)
                for name in classes:
                    c = float(per_class[CLASS_TO_ID[name]])
                    if c > best_confs[name]:
                        best_confs[name] = c
                scored += 1
                if any(c >= early_term for c in best_confs.values()):
                    break
            idx += 1
        cap.release()
        return best_confs, scored
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _matches_thresholds(confs, thresholds):
    """True if any class's cached conf >= its threshold."""
    return any(confs.get(name, 0.0) >= t for name, t in thresholds.items())


def _iter_all_assets(client):
    """Paginate /api/search/metadata over the user's full library."""
    page = 1
    while True:
        payload = {"page": page, "size": SEARCH_PAGE_SIZE, "withExif": False}
        r = client.session.post(
            f"{client.server_url}/api/search/metadata",
            json=payload,
            timeout=30,
        )
        r.raise_for_status()
        assets = r.json().get("assets", {})
        items = assets.get("items", [])
        if not items:
            return
        yield from items
        nxt = assets.get("nextPage")
        # Treat empty string / 0 / null as terminal, matching sidecar.py's
        # paginator. Immich sometimes returns "" instead of null at the end.
        if not nxt:
            return
        page = int(nxt)


def fetch_source_image_classifier(client, source, rule_name, logger):
    """Handler entry point. Signature matches SOURCE_HANDLERS plus rule_name +
    logger; sidecar.py passes both through. `logger` must accept `level=` kw
    (see the `log()` helper in sidecar.py).
    """
    thresholds = source.get("classes", {"cat": 0.30, "dog": 0.70})
    video_opts = {**DEFAULT_VIDEO_OPTS, **source.get("video", {})}
    model_path = source.get("model_path", DEFAULT_MODEL_PATH)
    classes = list(thresholds.keys())

    session = None  # lazy
    input_name = None
    expected_model = Path(model_path).stem
    cache = load_cache(rule_name, expected_model=expected_model, logger=logger)

    counts = {"cached": 0, "classified": 0, "errors": 0, "evicted": 0,
              "skipped_trashed": 0, "skipped_video_long": 0}
    matched_ids = set()
    seen_ids = set()
    unsaved = 0
    iter_completed = False

    try:
        for asset in _iter_all_assets(client):
            asset_id = asset["id"]
            seen_ids.add(asset_id)

            if asset.get("isTrashed") or asset.get("isArchived"):
                counts["skipped_trashed"] += 1
                if cache.pop(asset_id, None) is not None:
                    unsaved += 1
                continue

            cached = cache.get(asset_id)
            cached_confs = (cached or {}).get("confs", {})
            # Cache hit only if every class we currently care about has a cached score.
            if cached and all(c in cached_confs for c in classes):
                counts["cached"] += 1
                if _matches_thresholds(cached_confs, thresholds):
                    matched_ids.add(asset_id)
                continue

            kind = asset.get("type", "")
            try:
                if session is None:
                    session = load_session(model_path, prefer_gpu=False)
                    input_name = session.get_inputs()[0].name

                if kind == "IMAGE":
                    confs, frames = _classify_image(session, input_name, client, asset_id, classes)
                elif kind == "VIDEO":
                    duration_s = parse_duration(asset.get("duration"))
                    if duration_s > video_opts["max_duration_seconds"]:
                        counts["skipped_video_long"] += 1
                        cache[asset_id] = {
                            "kind": kind,
                            "classified_at": _now_iso(),
                            "skipped": "video_too_long",
                            "duration_s": duration_s,
                            "confs": {c: 0.0 for c in classes},
                        }
                        unsaved += 1
                        continue
                    confs, frames = _classify_video(
                        session, input_name, client, asset_id, classes, video_opts,
                    )
                else:
                    continue
            except Exception as e:
                counts["errors"] += 1
                logger(f"rule {rule_name}: classify {asset_id} failed: {e!r}", level="warn")
                continue

            cache[asset_id] = {
                "kind": kind,
                "classified_at": _now_iso(),
                "frames_scored": frames,
                "confs": {name: round(c, 4) for name, c in confs.items()},
            }
            unsaved += 1
            counts["classified"] += 1
            if _matches_thresholds(confs, thresholds):
                matched_ids.add(asset_id)

            if unsaved >= SAVE_EVERY:
                save_cache(rule_name, cache, model_path)
                unsaved = 0
        iter_completed = True
    finally:
        # Drop cache entries for assets no longer in the library, but only if
        # we successfully iterated the whole thing. A partial seen_ids would
        # incorrectly nuke valid rows from later pages.
        if iter_completed:
            stale = set(cache.keys()) - seen_ids
            if stale:
                for k in stale:
                    cache.pop(k, None)
                counts["evicted"] = len(stale)
                unsaved += len(stale)
        if unsaved > 0:
            save_cache(rule_name, cache, model_path)

    logger(
        f"rule {rule_name}: image_classifier "
        f"cached={counts['cached']} classified={counts['classified']} "
        f"errors={counts['errors']} evicted={counts['evicted']} "
        f"skip_trashed={counts['skipped_trashed']} "
        f"skip_vid_long={counts['skipped_video_long']} matched_total={len(matched_ids)}"
    )
    return matched_ids
