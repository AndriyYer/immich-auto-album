# immich-auto-album

A polling sidecar for [Immich](https://immich.app) that keeps shared albums populated based on rules. The only rule type today is "every photo containing a given face cluster", which I use to get something like Google Photos partner sharing - take a photo of your partner, it shows up in the album you share with them, and vice versa.

## How it works

Each cycle, for every rule:

1. Read current album members.
2. Anything we previously added that's now missing gets marked "removed" - if you delete a photo from the album in the UI, the sidecar won't add it back next time.
3. Anything previously marked removed that's back in the album gets unmarked.
4. Compute the desired set from the rule's source handler.
5. Add anything missing.

State lives at `data/state.json`.

## Setup

You need an Immich install reachable by hostname from another container on the same docker network, plus an API key per user the sidecar acts as. Minimum scopes: `asset.read`, `album.read`, `albumAsset.create`, `person.read`.

Drop the project into your immich compose dir (so it sits at `./auto-album/` next to `compose.yaml`), add the service block from `compose-snippet.yaml` to your compose file, then:

```
docker compose up -d auto-album
```

### Config

`config/rules.json` - copy `rules.example.json` and fill in real values. Each rule has:

- `name` - used for logs and state keys
- `api_key_env` - env var the API key lives in (one container can run rules for multiple users)
- `source.type` + `source.person_id` - face cluster UUID (find via `GET /api/people`)
- `target_album_id` - album to add to (create + share it in the UI first)

`.env` - whatever vars your `api_key_env` references. chmod 600.

The sidecar doesn't manage sharing on purpose. Creating and sharing in the UI keeps it explicit about who sees what.

## Adding a new source type

`SOURCE_HANDLERS` in `sidecar.py` is a dict keyed by `source.type`. Add a function with signature `(client, source_config) -> set[str]` of asset IDs and the rest is reused.

## Limitations

- Polling, not webhook. Immich doesn't expose a "new asset" event, so this is a 30 min loop by default. Tighten via `poll_interval_seconds` in rules.json.
- First run is unbounded. Point a rule at someone with 5000 tagged photos and all 5000 go in. Add a date cutoff in the handler if you don't want that.
- Smart search has no similarity threshold in the Immich API, so rule types that need confidence filtering have to do it client-side.
