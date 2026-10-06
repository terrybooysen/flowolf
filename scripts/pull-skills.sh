#!/usr/bin/env bash
# Pull Flowolf's live skills back from its server into this repo, so you can
# see (git diff) what the bot changed about itself and keep a history.
# Run from your Mac:
#   ./scripts/pull-skills.sh <ssh-target> <app-dir>
#   ./scripts/pull-skills.sh root@1.2.3.4 /docker/hermes-agent-xxxx
#
# Only pulls Flowolf's own skills (data/skills/flowolf), not the built-in
# Hermes skills. Never pulls .env files or credentials.
set -euo pipefail

TARGET=${1:?ssh target}
APP=${2:?app dir on server}
cd "$(dirname "$0")/.."

mkdir -p skills
ssh "$TARGET" "tar czf - -C '$APP/data/skills/flowolf' ." | tar xzf - -C skills
echo "Pulled $APP/data/skills/flowolf -> skills/"
git status --short skills
echo "Review with: git diff skills"
