# immich-auto-album

A polling sidecar for [Immich](https://immich.app) that keeps shared albums populated based on rules. Two rule types today:

- **`person`** - every photo containing a given face cluster. I use this for something like Google Photos partner sharing: take a photo of your partner, it shows up in the album you share with them, and vice versa.
- **`image_classifier`** - every photo or short video whose YOLO11m score for one of the configured COCO classes is above a per-class threshold. I use this for a shared "cats" album that picks up cat photos from both partners' libraries automatically.

## How it works

Each cycle, for every rule:

1. Read current album members.
2. Anything we previously added that's now missing gets marked "removed" - if you delete a photo from the album in the UI, the sidecar won't add it back next time.
3. Anything previously marked removed that's back in the album gets unmarked.
4. Compute the desired set from the rule's source handler.
5. Add anything missing.

State lives at `data/state.json`.

## Setup

You need an Immich install reachable by hostname from another container on the same docker network, plus an API key per user the sidecar acts as. Minimum scopes: `asset.read`, `album.read`, `albumAsset.create`, `person.read`. (The `image_classifier` source needs two more - see its section below.)

1. Clone this repo into your immich compose dir and rename so it sits at `./auto-album/` next to `compose.yaml`:

   ```
   git clone https://github.com/AndriyYer/immich-auto-album.git auto-album
   ```

   (Or clone wherever and adjust `build.context` in the compose snippet.)

2. Append the service block from `compose-snippet.yaml` to your `compose.yaml` under `services:`.

3. Create the config dir and copy the example rules file, then edit it:

   ```
   mkdir -p ./auto-album/config ./auto-album/data
   cp ./auto-album/rules.example.json ./auto-album/config/rules.json
   $EDITOR ./auto-album/config/rules.json
   ```

4. Create `./auto-album/.env` with one line per API key your rules reference (env var name matches `api_key_env` in each rule) and lock it down:

   ```
   ALICE_API_KEY=...
   BOB_API_KEY=...
   ```

   `chmod 600 ./auto-album/.env`

5. Build and start:

   ```
   docker compose up -d --build auto-album
   ```

The sidecar doesn't manage album creation or sharing on purpose. Create the target albums in the Immich UI first and share them with whomever you want; the sidecar only adds assets.

### `rules.json` fields

Top-level:

- `server_url` - how the sidecar reaches Immich (`http://immich-server:2283` if on the same docker network as the standard compose).
- `poll_interval_seconds` - cycle interval, default 1800 (30 min).
- `rules` - list of rule objects below.

Per rule:

- `name` - used for logs and state keys; must be unique.
- `api_key_env` - name of the env var holding the Immich API key for this rule (so one container can run rules for multiple users).
- `source.type` - `person` or `image_classifier`.
- `source.person_id` (person rules) - face cluster UUID. Find with `curl -H "x-api-key: $KEY" https://<your-immich>/api/people`.
- `source.classes`, `source.video` (image_classifier rules) - see below.
- `target_album_id` - album UUID to add into. Grab it from the album's URL after you create it in the UI.

### `image_classifier` source

Requires two extra scopes on the API key: `asset.view` (thumbnail + transcoded video) and `asset.download` (original video fallback).

Source schema:

```json
{
  "type": "image_classifier",
  "classes": {"cat": 0.30, "dog": 0.70},
  "video": {
    "max_duration_seconds": 300,
    "max_frames": 20,
    "sample_period_seconds": 2.0,
    "early_terminate_conf": 0.70
  }
}
```

`classes` maps COCO class names to minimum confidence. An asset matches if **any** class confidence is above its threshold (i.e. dog at 0.70 is a useful co-trigger for misclassified cats). `video` controls how long videos are sampled before giving up.

You also need the YOLO11m ONNX model on disk:

```
mkdir -p ./auto-album/models
# Download yolo11m.pt from https://github.com/ultralytics/assets/releases
# then export to ONNX:
pip install ultralytics
yolo export model=yolo11m.pt format=onnx opset=12
mv yolo11m.onnx ./auto-album/models/yolo11m.onnx
```

(Any YOLO11/v8-format ONNX model will work; the handler reads `(1, 4+nc, num_anchors)` output. Smaller models run faster but recall drops.)

Per-rule confidence cache lives at `./auto-album/data/classifier_cache/<rule>.json`. The cache stores raw per-class confidences, not match decisions, so changing thresholds in `rules.json` doesn't invalidate cached work. Delete the file to force reclassification.

### Optional: seed the cache from a GPU host

YOLO11m on a low-power CPU is ~2s/asset, so the sidecar's first full scan of a 20K-asset library would take ~12 hours. If you have access to a GPU box (CUDA), run `classifier/backlog.py` there to do the one-shot pass, then copy the cache over so the sidecar starts seeded:

```
# On the GPU host:
git clone https://github.com/AndriyYer/immich-auto-album.git
cd immich-auto-album
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt onnxruntime-gpu
mkdir classifier/models && cp /path/to/yolo11m.onnx classifier/models/

export MY_API_KEY=...
python -m classifier.backlog \
  --server https://<your-immich> \
  --api-key-env MY_API_KEY \
  --album-id <target-album-uuid> \
  --label me

# Add --skip-album-write for a dry run that produces just the CSV.

# Then convert the CSV into a cache file the sidecar can consume:
python -m classifier.cache_from_csv \
  classifier/logs/run-me-<timestamp>.csv \
  /tmp/me.json --classes cat dog

# Copy /tmp/me.json into your sidecar host at:
#   ./auto-album/data/classifier_cache/<rule-name>.json
# where <rule-name> matches the rule's `name` field in rules.json.
```

## Adding a new source type

`SOURCE_HANDLERS` in `sidecar.py` is a dict keyed by `source.type`. Add a function with signature `(client, source_config, rule_name=None, logger=None) -> set[str]` of asset IDs and the rest is reused.

## Limitations

- Polling, not webhook. Immich doesn't expose a "new asset" event, so this is a 30 min loop by default. Tighten via `poll_interval_seconds` in rules.json.
- First run is unbounded. Point a rule at someone with 5000 tagged photos and all 5000 go in. Add a date cutoff in the handler if you don't want that.
- `image_classifier` on a low-power host will be CPU-bound on the first full scan. Use the backlog seeder above if that matters.

## License

MIT. See `LICENSE`.
