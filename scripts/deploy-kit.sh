#!/usr/bin/env bash
# Deploy Flowolf's kit (rules, skills, tools) from this repo to its Hermes server.
# Run from your Mac:
#   ./scripts/deploy-kit.sh <ssh-target> <app-dir>
#   ./scripts/deploy-kit.sh root@1.2.3.4 /docker/hermes-agent-xxxx
#
# What it does (on the server):
#   - backs up data/SOUL.md, then replaces the flowolf rules block in it
#     (between markers, so it is safe to run again) with hermes/SOUL-addendum.md
#   - copies skills/ into data/skills/flowolf/ and tools/ into data/flowolf/
#   - never touches .env files or credentials
# Parts that don't exist in the repo yet are skipped.
set -euo pipefail

TARGET=${1:?ssh target, e.g. root@1.2.3.4}
APP=${2:?app dir on server, e.g. /docker/hermes-agent-xxxx}
BOT=flowolf
cd "$(dirname "$0")/.."

STAGE=$(mktemp -d)
mkdir -p "$STAGE/kit"
if [ -f hermes/SOUL-addendum.md ]; then cp hermes/SOUL-addendum.md "$STAGE/kit/"; fi
if [ -d skills ]; then cp -R skills "$STAGE/kit/skills"; fi
if [ -d tools ]; then cp -R tools "$STAGE/kit/tools"; fi
tar czf "$STAGE/kit.tgz" -C "$STAGE" kit
scp -q "$STAGE/kit.tgz" "$TARGET:/tmp/kit-$BOT.tgz"
rm -rf "$STAGE"

ssh "$TARGET" BOT="$BOT" APP="$APP" 'bash -s' <<'REMOTE'
set -euo pipefail
D="$APP/data"
W=$(mktemp -d)
tar xzf "/tmp/kit-$BOT.tgz" -C "$W"
K="$W/kit"

# 1. Rules block in SOUL.md (backup first, replace between markers)
if [ -f "$K/SOUL-addendum.md" ]; then
  cp "$D/SOUL.md" "$D/SOUL.md.bak-$(date +%F-%H%M%S)"
  python3 - "$D/SOUL.md" "$K/SOUL-addendum.md" "$BOT" <<'PY'
import re, sys
soul_path, add_path, bot = sys.argv[1:4]
soul = open(soul_path).read()
add = open(add_path).read().strip()
begin, end = f"<!-- BEGIN {bot} rules -->", f"<!-- END {bot} rules -->"
soul = re.sub(re.escape(begin) + r".*?" + re.escape(end), "", soul, flags=re.S)
soul = soul.rstrip() + f"\n\n{begin}\n{add}\n{end}\n"
open(soul_path, "w").write(soul)
PY
fi

# 2. Skills -> data/skills/flowolf/
if [ -d "$K/skills" ]; then
  mkdir -p "$D/skills/$BOT"
  cp -R "$K/skills/." "$D/skills/$BOT/"
fi

# 3. Tools -> data/flowolf/ (never overwrites .env)
if [ -d "$K/tools" ]; then
  mkdir -p "$D/$BOT/backups"
  cp -R "$K/tools/." "$D/$BOT/"
  touch "$D/$BOT/WORKLOG.md"
  chmod 700 "$D/$BOT"
fi

chown -R 10000:10000 "$D/SOUL.md" "$D/skills/$BOT" "$D/$BOT" 2>/dev/null || true
rm -rf "$W" "/tmp/kit-$BOT.tgz"
echo "DEPLOYED $BOT -> $D"
REMOTE

echo "Done. In Telegram, send the bot /reset so it reloads its rules."
