"""
Fetch (or check) the raw data the paper uses.

  python scripts/data/fetch_sources.py --list
  python scripts/data/fetch_sources.py            # download what can be downloaded
  python scripts/data/fetch_sources.py --check    # verify only, no network

Sources with no direct URL (Retro* on Dropbox, the MSU property datasets) are
reported with instructions instead of being guessed at. Everything that lands
is recorded in data/raw/manifest.json with its size and sha256, so a later run
can tell a complete download from a truncated one.
"""
import argparse
import json
import os
import sys
import urllib.request

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sources import RAW_DIR, SOURCES, describe, sha256

MANIFEST = os.path.join(RAW_DIR, "manifest.json")


def load_manifest():
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            return json.load(f)
    return {}


def save_manifest(manifest):
    os.makedirs(RAW_DIR, exist_ok=True)
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024


def download(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    print(f"  downloading {url}")

    def hook(blocks, block_size, total):
        if total > 0 and blocks % 200 == 0:
            done = blocks * block_size
            print(f"\r  {human(done)} / {human(total)}", end="", flush=True)

    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    print()
    os.replace(tmp, dest)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true", help="describe the sources and exit")
    p.add_argument("--check", action="store_true", help="verify what is on disk, no network")
    p.add_argument("--only", nargs="+", help="source names to act on")
    p.add_argument("--force", action="store_true", help="re-download even if present")
    args = p.parse_args()

    sources = [s for s in SOURCES if not args.only or s.name in args.only]

    if args.list:
        for s in sources:
            print(describe(s))
            print()
        return

    manifest = load_manifest()
    missing = []

    for s in sources:
        present = os.path.exists(s.path)
        is_dir = present and os.path.isdir(s.path)

        if present and not args.force:
            if is_dir:
                files = sorted(os.listdir(s.path))
                print(f"[{s.name}] directory with {len(files)} file(s): {', '.join(files[:6])}")
                if not files:
                    missing.append(s)
            else:
                size = os.path.getsize(s.path)
                print(f"[{s.name}] present, {human(size)}")
                entry = manifest.get(s.name, {})
                if entry.get("size") != size or "sha256" not in entry:
                    manifest[s.name] = {"path": s.path, "size": size, "sha256": sha256(s.path)}
            continue

        if args.check:
            print(f"[{s.name}] MISSING -> {s.path}")
            missing.append(s)
            continue

        if s.url:
            download(s.url, s.path)
            size = os.path.getsize(s.path)
            manifest[s.name] = {"path": s.path, "size": size, "sha256": sha256(s.path),
                                "url": s.url}
            print(f"[{s.name}] done, {human(size)}")
        else:
            print(f"[{s.name}] needs a manual download:")
            print(describe(s))
            missing.append(s)

    save_manifest(manifest)

    if missing:
        print("\nStill missing:", ", ".join(s.name for s in missing))
        print("Run with --list for the instructions.")
        sys.exit(1)


if __name__ == "__main__":
    main()
