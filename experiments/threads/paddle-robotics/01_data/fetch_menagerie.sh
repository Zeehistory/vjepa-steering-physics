#!/usr/bin/env bash
# Fetch the Franka Emika Panda model used by src/data/franka_strike.py.
#
# Vendors mujoco_menagerie's franka_emika_panda (~36 MB of meshes) BESIDE the repo rather than into
# it, so it is shared across branches and never enters git history. Override the destination with
# MENAGERIE_DIR; franka_strike.py checks $MENAGERIE_DIR first, then this default location.
#
# Idempotent: re-running with the model already present is a no-op.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${MENAGERIE_DIR:-$(dirname "$REPO")/assets/mujoco_menagerie}"

if [[ -f "$DEST/franka_emika_panda/panda_nohand.xml" ]]; then
  echo "Panda already present at $DEST/franka_emika_panda -- nothing to do."
  exit 0
fi

echo "Fetching franka_emika_panda into $DEST ..."
mkdir -p "$(dirname "$DEST")"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Sparse + blobless: pulls only the Panda directory, not the whole ~1 GB menagerie.
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/google-deepmind/mujoco_menagerie.git "$TMP/mm"
git -C "$TMP/mm" sparse-checkout set franka_emika_panda

mkdir -p "$DEST"
cp -r "$TMP/mm/franka_emika_panda" "$DEST/"
cp "$TMP/mm/LICENSE" "$DEST/" 2>/dev/null || true

echo "Done: $DEST/franka_emika_panda"
echo "Verify with:  PYTHONPATH=. python -c 'from src.data.franka_strike import FrankaStrike; FrankaStrike()'"
