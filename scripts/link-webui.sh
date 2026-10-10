#!/usr/bin/env bash
# Link Open WebUI to Flowolf's Hermes when both run on the same server.
# Run from your Mac (asks for the server's root password unless your SSH key is on it):
#   cd ~/flowolf && git pull && ssh root@<server-ip> 'bash -s' < scripts/link-webui.sh
#
# What it does on the server:
#   - stops if Open WebUI still lets anyone sign up (an account could then run
#     commands on the server through Hermes)
#   - switches on Hermes's API server, reusing its key if one is already set
#   - puts both apps on a shared Docker network, flowolf-net
#   - restarts the Hermes gateway (Telegram drops for a few seconds)
#   - checks Open WebUI can reach Hermes, then prints the URL and key to enter
#     in Open WebUI (Admin Panel > Settings > Connections > OpenAI API > +)
# Safe to run again. If Hostinger updates either app, run it again.
set -uo pipefail

say() { printf '%s\n' "$*"; }
die() { printf '\nSTOPPED: %s\n' "$*"; exit 1; }

names=$(docker ps --format '{{.Names}}')
H=$(printf '%s\n' "$names" | grep -m1 'hermes-agent' || true)
W=$(printf '%s\n' "$names" | grep -m1 -i 'webui' || true)
[ -n "$H" ] || die "no running hermes-agent container. Running: $(echo $names)"
[ -n "$W" ] || die "no running Open WebUI container. Running: $(echo $names)"
say "Hermes:     $H"
say "Open WebUI: $W"

# Fetch a URL from inside the Open WebUI container (optional bearer key in $K).
fetch_w() {
  docker exec -e K="${2:-}" "$W" python3 -c '
import os, sys, urllib.request
req = urllib.request.Request(sys.argv[1])
if os.environ.get("K"):
    req.add_header("Authorization", "Bearer " + os.environ["K"])
print(urllib.request.urlopen(req, timeout=5).read().decode())' "$1" 2>/dev/null
}

# 1. Refuse to link while anyone can sign up.
cfg=$(fetch_w "http://localhost:8080/api/config" || true)
if printf '%s' "$cfg" | grep -q '"enable_signup": *true'; then
  die "Open WebUI still allows new sign-ups. Admin Panel > Settings > General > switch off Enable New Sign Ups, then run this again."
elif printf '%s' "$cfg" | grep -q '"enable_signup": *false'; then
  say "Sign-ups:   off"
else
  say "WARNING: couldn't read Open WebUI's sign-up setting. Make sure Enable New Sign Ups is off."
fi

# 2. Hermes API server, reusing an existing key.
KEY=$(docker exec "$H" bash -lc 'grep -hs "^API_SERVER_KEY=" "${HERMES_HOME:-$HOME/.hermes}/.env" /opt/data/.env' 2>/dev/null \
  | tail -1 | cut -d= -f2- | tr -d '\r"'"'" || true)
if [ -n "$KEY" ]; then
  say "API key:    reusing the existing one"
else
  KEY=$(docker exec "$H" python3 -c 'import secrets; print(secrets.token_hex(24))') || die "couldn't generate an API key"
  say "API key:    new"
fi
docker exec -e K="$KEY" "$H" bash -lc 'hermes config set API_SERVER_ENABLED true && hermes config set API_SERVER_HOST 0.0.0.0 && hermes config set API_SERVER_KEY "$K"' >/dev/null \
  || die "hermes config set failed inside $H"

# 3. Shared network.
docker network inspect flowolf-net >/dev/null 2>&1 || docker network create flowolf-net >/dev/null || die "couldn't create network flowolf-net"
for c in "$H" "$W"; do
  docker inspect -f '{{json .NetworkSettings.Networks}}' "$c" | grep -q '"flowolf-net"' \
    || docker network connect flowolf-net "$c" || die "couldn't connect $c to flowolf-net"
done
say "Network:    flowolf-net"

# 4. Restart the gateway so the API server starts with the new settings.
docker exec "$H" /package/admin/s6/command/s6-svc -r /run/service/gateway-default || die "gateway restart failed"

# 5. Check Open WebUI can reach Hermes and sees the model.
ok=""
for _ in $(seq 1 15); do
  sleep 2
  if fetch_w "http://$H:8642/health" | grep -q '"ok"'; then ok=1; break; fi
done
if [ -z "$ok" ]; then
  say ""
  say "Gateway log, API server lines:"
  docker exec "$H" bash -lc 'grep -i "api server" /opt/data/logs/gateway.log 2>/dev/null | tail -5' || true
  die "Open WebUI can't reach Hermes's API server. Paste this output to Claude."
fi
fetch_w "http://$H:8642/v1/models" "$KEY" | grep -q '"id"' || die "API server answers, but the key was rejected. Paste this output to Claude."

say ""
say "LINKED. In Open WebUI: Admin Panel > Settings > Connections > OpenAI API > +"
say "  URL: http://$H:8642/v1"
say "  Key: $KEY"
say "Save, switch the Ollama connection off, then pick hermes-agent in a new chat."
