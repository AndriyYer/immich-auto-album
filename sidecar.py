import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", "/app/config/rules.json"))
STATE_PATH = Path(os.environ.get("STATE_PATH", "/app/data/state.json"))
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "30"))
SEARCH_PAGE_SIZE = 250
ALBUM_ADD_CHUNK = 500

_shutdown = False


def log(msg, level="info"):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"{ts} [{level}] {msg}", flush=True)


def handle_signal(signum, _frame):
    global _shutdown
    log(f"caught signal {signum}, will exit after current cycle")
    _shutdown = True


class ImmichClient:
    def __init__(self, server_url, api_key):
        self.server_url = server_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "x-api-key": api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        })

    def _request(self, method, path, **kwargs):
        url = f"{self.server_url}{path}"
        resp = self.session.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
        resp.raise_for_status()
        return resp.json() if resp.content else None

    def get_album_assets(self, album_id):
        data = self._request("GET", f"/api/albums/{album_id}")
        return {a["id"] for a in data.get("assets", [])}

    def search_by_person(self, person_id):
        ids = set()
        page = 1
        while True:
            payload = {
                "personIds": [person_id],
                "page": page,
                "size": SEARCH_PAGE_SIZE,
                "withExif": False,
            }
            data = self._request("POST", "/api/search/metadata", json=payload)
            assets = data.get("assets", {})
            for item in assets.get("items", []):
                ids.add(item["id"])
            next_page = assets.get("nextPage")
            if not next_page:
                break
            page = int(next_page)
        return ids

    def add_to_album(self, album_id, asset_ids):
        added = 0
        asset_list = list(asset_ids)
        for i in range(0, len(asset_list), ALBUM_ADD_CHUNK):
            chunk = asset_list[i:i + ALBUM_ADD_CHUNK]
            result = self._request("PUT", f"/api/albums/{album_id}/assets", json={"ids": chunk})
            for item in result or []:
                if item.get("success"):
                    added += 1
        return added


def fetch_source_person(client, source):
    return client.search_by_person(source["person_id"])


SOURCE_HANDLERS = {
    "person": fetch_source_person,
}


def load_state():
    if not STATE_PATH.exists():
        return {}
    try:
        with STATE_PATH.open() as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log(f"failed to read state file, starting fresh: {e}", level="warn")
        return {}
    return {
        name: {"added": set(s.get("added", [])), "removed": set(s.get("removed", []))}
        for name, s in raw.items()
    }


def save_state(state):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        name: {"added": sorted(s["added"]), "removed": sorted(s["removed"])}
        for name, s in state.items()
    }
    tmp = STATE_PATH.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(serializable, f, indent=2)
    tmp.replace(STATE_PATH)


def load_config():
    with CONFIG_PATH.open() as f:
        return json.load(f)


def run_rule(rule, server_url, rule_state):
    api_key = os.environ.get(rule["api_key_env"])
    if not api_key:
        log(f"rule {rule['name']}: env var {rule['api_key_env']} not set, skipping", level="error")
        return rule_state

    source_type = rule["source"]["type"]
    handler = SOURCE_HANDLERS.get(source_type)
    if handler is None:
        log(f"rule {rule['name']}: unknown source type {source_type!r}, skipping", level="error")
        return rule_state

    client = ImmichClient(server_url, api_key)
    album_id = rule["target_album_id"]

    try:
        current = client.get_album_assets(album_id)
    except requests.RequestException as e:
        log(f"rule {rule['name']}: failed to read album {album_id}: {e}", level="error")
        return rule_state

    previously_added = rule_state["added"]
    removed = rule_state["removed"]

    new_removals = previously_added - current
    if new_removals:
        log(f"rule {rule['name']}: {len(new_removals)} assets removed manually, will not re-add")
        removed = removed | new_removals

    restored = removed & current
    if restored:
        log(f"rule {rule['name']}: {len(restored)} previously removed assets are back, clearing from removed list")
        removed = removed - restored

    try:
        desired = handler(client, rule["source"])
    except requests.RequestException as e:
        log(f"rule {rule['name']}: source fetch failed: {e}", level="error")
        return {"added": current - removed, "removed": removed}

    to_add = desired - current - removed
    if to_add:
        log(f"rule {rule['name']}: adding {len(to_add)} new assets to album")
        try:
            added_count = client.add_to_album(album_id, to_add)
            log(f"rule {rule['name']}: added {added_count}/{len(to_add)}")
        except requests.RequestException as e:
            log(f"rule {rule['name']}: add_to_album failed: {e}", level="error")
            return {"added": current - removed, "removed": removed}
        current = current | to_add
    else:
        log(f"rule {rule['name']}: no new matches (desired={len(desired)}, current={len(current)}, suppressed={len(removed)})")

    return {"added": current - removed, "removed": removed}


def main():
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    log(f"starting; config={CONFIG_PATH}, state={STATE_PATH}")
    cfg = load_config()
    interval = int(cfg.get("poll_interval_seconds", 1800))
    server_url = cfg["server_url"]
    rules = cfg["rules"]
    log(f"loaded {len(rules)} rules, polling every {interval}s against {server_url}")

    while not _shutdown:
        state = load_state()
        for rule in rules:
            rule_state = state.get(rule["name"], {"added": set(), "removed": set()})
            try:
                state[rule["name"]] = run_rule(rule, server_url, rule_state)
            except Exception as e:
                log(f"rule {rule['name']}: unexpected error: {e}", level="error")
        save_state(state)

        if _shutdown:
            break

        slept = 0
        while slept < interval and not _shutdown:
            time.sleep(min(5, interval - slept))
            slept += 5

    log("shutting down cleanly")


if __name__ == "__main__":
    main()
