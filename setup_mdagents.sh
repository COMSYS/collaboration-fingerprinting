#!/usr/bin/env bash
# Obtain upstream MDAgents. It is third-party code and is NOT redistributed in
# this repository; this script fetches it from its own source.
#
# The commit is pinned: the in-memory corrections in
# mdagents_sidechannel/upstream.py match this revision exactly and refuse to
# run against anything else.
set -euo pipefail

UPSTREAM_URL="https://github.com/mitmedialab/MDAgents.git"
UPSTREAM_SHA="3adbd760ca809b4e7b0c1085d68314b6e7d91e1b"
DEST="${MDAGENTS_HOME:-$(cd "$(dirname "$0")" && pwd)/third_party/MDAgents}"

if [ -d "$DEST/.git" ]; then
  echo "[setup] existing checkout at $DEST"
else
  echo "[setup] cloning $UPSTREAM_URL -> $DEST"
  mkdir -p "$(dirname "$DEST")"
  git clone --quiet "$UPSTREAM_URL" "$DEST"
fi

echo "[setup] checking out pinned commit $UPSTREAM_SHA"
git -C "$DEST" fetch --quiet origin
git -C "$DEST" checkout --quiet "$UPSTREAM_SHA"

PY="${PYTHON:-python3}"
PYVER="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "[setup] python: $PYVER ($PY)"
"$PY" - <<'PYCHECK'
import sys
if sys.version_info < (3, 12):
    sys.exit(
        f"Python 3.12+ required (found {sys.version_info.major}.{sys.version_info.minor}).\n"
        "Upstream utils.py has a backslash inside an f-string expression, which only\n"
        "parses from 3.12 on (PEP 701)."
    )
PYCHECK

echo "[setup] verifying the corrections still match upstream"
cd "$(dirname "$0")"
MDAGENTS_HOME="$DEST" "$PY" -c "
from pathlib import Path
from mdagents_sidechannel.upstream import apply_fixes, mdagents_home
src = (mdagents_home() / 'utils.py').read_text()
_, applied = apply_fixes(src)
print(f'[setup] {len(applied)} corrections apply cleanly')
"

echo "[setup] done. Upstream MDAgents is at: $DEST"
echo "[setup] export MDAGENTS_HOME=$DEST  (or leave unset to use this default)"
