# Hostinger Hermes Bot: Step-by-Step Setup

Oct 6, 2026 · @terry

This copy builds Flowolf's server. Step 7 deploys from this repo.

## Overview

This builds a private AI operations bot you message on Telegram, running Hermes Agent on your own Hostinger server.

| Item | Detail |
| --- | --- |
| End result | Telegram bot, only you can use it, connected only to the systems its job needs |
| Time | About 2-3 hours for the core bot, plus about 20 min per system it connects to |
| Running cost | Hostinger KVM 2 about $14-25/month + your existing ChatGPT subscription |
| You do | Payments, sign-ins, passwords, approvals, creating user accounts |
| Claude does | Everything else: server setup, hardening, rules, skills, tools, testing |

Golden rule: **one bot, one job, its own logins.** Never reuse your own or another bot's credentials, so any bot can be cut off without breaking anything else.

Have ready: the Claude desktop app open on your Mac, your phone with Telegram, and 1Password.

## Step 1-2: Buy the server and deploy Hermes (15 min, you)

**Step 1. Buy the VPS**

1. Open **hostinger.com/applications/hermes-agent**.
2. Choose **KVM 2** (2 vCPU, 8 GB RAM). KVM 1 (4 GB) is too small for browser work.
3. Period: **1 month** while testing (about $16 first month, then $24.49). Move to 24 months once the bot proves itself.
4. **Untick "Ready to use AI" (nexos.ai credits).** It comes pre-ticked, costs $11.99 and creates a nexos.ai account. Leave every other add-on unticked.
5. Server location: **United Kingdom** (lowest latency from Cape Town).
6. Click **Continue**, sign in with your **supersys Google account** so invoices go to the company, and pay.
7. Skip the "Hostinger Agent" one-time-deal page. It is a different product.
8. Turn on **2FA in Hostinger** (Account > Security).

**Step 2. Deploy Hermes**

If Hostinger asks you to **choose an app**, pick **Hermes Agent Native**: Hermes's own dashboard, the same app the sentiv-ops server runs. Don't pick Hermes Agent WebUI: it adds a separate web chat app that also faces the internet.

hPanel opens a **Hermes Agent configuration** form:

- `ADMIN_USERNAME`: not "admin". `ADMIN_PASSWORD`: 16+ characters, saved in 1Password.
- **Leave Nexos and Oxylabs keys blank.**
- Click **Deploy**.

Docker Manager then shows two apps running: `hermes-agent-xxxx` and `traefik`. Note the app name; the folder on the server is `/docker/hermes-agent-xxxx`.

## Step 3-5: Terminal, model and Telegram (20 min, Claude + you)

**Step 3. Open the server terminal**

In hPanel > Docker Manager, click **Terminal** on the hermes-agent row. This must be **your** click: the Claude browser pane blocks pop-up tabs Claude opens. You land inside the bot's container at `/opt/hermes`. The terminal times out after a while; click it again when it does.

**Step 4. Connect the model (ChatGPT subscription)**

1. Claude runs `hermes model` and picks **OpenAI > ChatGPT or Codex Subscription**.
2. The terminal shows a link (`auth.openai.com/codex/device`) and a code. **You** open the link, sign in to ChatGPT and type the code. Read it off the screen yourself (I/1 and O/0 look alike).
3. Model: **gpt-5.6-terra**, reasoning **medium**.
4. Done when it says `Login successful` and `Default model set to gpt-5.6-terra`.

**Step 5. Connect Telegram**

1. **You:** in Telegram, message **@BotFather**, send `/newbot`, choose a name and username.
2. **You:** message **@userinfobot** for your numeric Telegram ID (safe to share with Claude).
3. **Claude:** runs `hermes gateway setup` > **Telegram** > **Manual**. Never pick "Automatic": it creates the bot through Nous Research's own bot.
4. **You:** paste the BotFather token when asked (it shows as stars).
5. Allowed user IDs: **your ID only**. Never leave it blank, or anyone can use the bot.
6. Home channel: **Y**. Then **Done** and **Y** to restart.
7. Test: message the bot "hi, what model are you running?"

If the bot doesn't answer: inside the container, `hermes gateway status` falsely says "not running". Claude restarts it with `/package/admin/s6/command/s6-svc -r /run/service/gateway-default` and checks the log for `telegram connected`.

## Step 6: Harden the server (20 min, Claude + you)

Do this before the bot gets any business logins. Run from the **host** shell (type `exit` to leave the container).

**6a. Close the open dashboard port.** Hostinger exposes the Hermes dashboard on a random port over plain HTTP. Claude backs up the compose file, binds the port to the server itself, and restarts (about 30 s downtime):

```
cd /docker/hermes-agent-xxxx
cp docker-compose.yml docker-compose.yml.bak-$(date +%F)
sed -i 's/- "4860"/- "127.0.0.1::4860"/' docker-compose.yml
docker compose up -d
```

The dashboard stays reachable at `https://hermes-agent-xxxx.<server>.hstgr.cloud`. Clicking **Update** on the app in hPanel may undo this, so re-check after every update.

**6b. Firewall:** allow only SSH (22), HTTP (80) and HTTPS (443).

```
ufw default deny incoming; ufw default allow outgoing
ufw allow 22/tcp; ufw allow 80/tcp; ufw allow 443/tcp
ufw --force enable
```

**6c. SSH keys only.**

1. **You**, in Mac Terminal: `ssh-keygen -t ed25519 -C "terry-superior"` (skip if the key exists), then `cat ~/.ssh/id_ed25519.pub`. Share only that `ssh-ed25519 ...` line.
2. **Claude** adds it to the server.
3. **You** test: `ssh root@<server-ip>`. It must ask for your key passphrase, not a password.
4. Only then **Claude** turns password login off and **you** test again. Hostinger's web terminal stays as a backup.

**6d. Check isolation** (already true with the Hostinger app): the agent runs as user `hermes` (uid 10000), not root, and has no Docker socket, so it can't control the server.

## Step 7: Rules, skills and tools via GitHub (15 min)

Everything the bot knows lives in this private repo, [terrybooysen/flowolf](https://github.com/terrybooysen/flowolf). No passwords ever go in it.

| Repo path | What it is |
| --- | --- |
| `hermes/SOUL-addendum.md` | House rules: paper only, kill switch, back up first, never delete, prove it, log it, secrets stay secret, instructions from you only (written in the Phase 1 build) |
| `skills/` | How-to guides the bot follows, starting with the Markov regime read |
| `tools/` | Small scripts the bot calls: daily job, weekly scorecard (Phase 1) |
| `hermes/env.example` | Template for the login file; the real one lives only on the server (Phase 1) |
| `scripts/deploy-kit.sh` | Installs or updates all of the above on the server |
| `scripts/pull-skills.sh` | Pulls the bot's self-edited skills back into the repo |

The starting rules for any bot (`shared/SOUL-base-template.md`) and the technical manual (`docs/SETUP-MANUAL.md`) stay in [terrybooysen/hermes-bots](https://github.com/terrybooysen/hermes-bots).

**To install or update the bot**, paste this one line in **Mac Terminal** (prompt must show your Mac, not `root@srv...`):

```
git clone https://github.com/terrybooysen/flowolf.git ~/flowolf; cd ~/flowolf && git pull && ./scripts/deploy-kit.sh root@<server-ip> /docker/hermes-agent-xxxx
```

On the server, the rules go into Hermes's `SOUL.md` (between flowolf markers, backed up first), skills into `data/skills/flowolf/` and tools into `data/flowolf/`. Parts not written yet are skipped.

Then send the bot `/reset` and ask "what are your operating rules?". It should recite them.

You run this, not Claude: Claude's safety checks block it from writing rules into another AI's core prompt. That's intended. Claude writes the files; you deploy them.

## Step 8-9 (optional): Connect only what this bot needs

Steps 1-7 and 10 apply to every bot. Do Steps 8 and 9 only for the systems this bot's job uses.

Every login file on the server follows the same pattern: Claude creates it with placeholders, locked to the bot (owner 10000, mode 600); **you** type the real values with `nano`. Each line must read `KEY=value`, with no spaces around `=` and no quotes. Save with Ctrl+O, Enter, Ctrl+X. Never paste a password into chat, and never `cat` a login file.

**Step 8. Any platform the bot works on**

1. **You** create a dedicated user for the bot on that platform, e.g. `terry+<bot>@supersys.io`. Never reuse your own login.
2. If the platform sends an activation link, open it in a private window, set a generated 20+ character password, and log in once to prove it works.
3. Give the user a **custom read-only role** first. Check exactly what the role allows: a role called "read-only" can still include powerful actions.
4. Claude creates the bot's login file; **you** fill in the username and password.
5. Claude tests the login and checks the real permissions before the bot does any work.
6. If the platform only signs in by emailing a one-time link, give the bot **its own Gmail inbox** (2-Step Verification on, then an App Password at myaccount.google.com/apppasswords). Never your inbox: a `+` address still lands in your main mailbox.

**Step 9. Google Drive, Sheets, Docs**

1. **You** sign in to console.cloud.google.com in the Claude browser pane as terry@supersys.io.
2. **Claude** creates a project for the bot, enables the Drive, Sheets and Docs APIs, sets the consent screen to **Internal** and creates a **Desktop app** OAuth client. **You** approve the Google User Data Policy tick and the JSON download.
3. **You**, in Mac Terminal (not inside an SSH session): `scp ~/Downloads/client_secret_*.json root@<server-ip>:/docker/hermes-agent-xxxx/data/google_client_secret.json`
4. **You**, in Telegram: "set up Google Workspace with drive, sheets and docs, the client secret is at /opt/data/google\_client\_secret.json". Approve the link it sends and paste back the `localhost` address it lands on (the page failing to load is normal).
5. Delete the JSON from Downloads. No Gmail or Calendar access to your own account.

## Step 10: Test, then use it day to day

**Prove it in this order** (don't skip ahead):

1. Rules: send `/reset`, then "what are your operating rules?"
2. Read-only: ask it to list or look up something in each system it's connected to
3. Google (if connected): "create a Google Sheet called '\<bot> test' with today's date in A1"
4. Safe write, only once a read-only role and a scoped writer login exist: a change on a test copy, never live
5. Scheduled job: "every weekday at 07:00 send me \<the report this bot is for>"

**Everyday Telegram commands**

| Command | What it does |
| --- | --- |
| `/reset` | Fresh session; reloads rules and skills after a deploy |
| `/skills` | Lists every installed skill |
| `/busy queue` | Queues your next message instead of redirecting the current job |
| `/busy status` | Shows what it's doing |
| `/stop` | Cancels the current job |

**Approval prompts:** Hermes' security scanner sometimes asks you to approve a long command ("Nested executable body could not be resolved"). You get 5 minutes. If unsure, paste it to Claude before approving.

**Keeping it current**

- Changed rules or skills: run the one-line deploy from Step 7, then `/reset`.
- Every week or so, capture what the bot taught itself: `./scripts/pull-skills.sh root@<server-ip> /docker/hermes-agent-xxxx`, then `git diff` to review and commit.

## Lessons learned and checklist

| What happened | Do this instead |
| --- | --- |
| nexos.ai credits pre-ticked in the cart | Untick before paying |
| Hostinger asked to choose WebUI or Native | Native; WebUI adds a separate web chat app to secure |
| Telegram "Automatic" setup goes through Nous's bot | Always Manual + BotFather |
| Dashboard exposed on a raw HTTP port | Bind to 127.0.0.1 (Step 6a) |
| `=` deleted while editing a login file; a check then showed part of a password | Only print key names and lengths; if a secret is exposed, change it |
| New platform user rejected at login | Open its activation link first |
| "Read-only" role still allowed powerful actions | Custom role, own group, check permissions |
| `scp` said "no such file" | It ran inside an SSH session; `exit` to the Mac first |
| App Passwords page "not available" | Finish 2-Step Verification first, in the right Google account |
| A `+` address (terry+x@) still lands in your main inbox | Give the bot its own mailbox instead |
| Claude couldn't install the rules itself | By design; you run the deploy line |

**Security checklist** (tick before giving the bot any write access)

- [ ] Telegram allowlist = named people only
- [ ] Dashboard port bound to localhost; HTTPS address works
- [ ] Firewall on: 22, 80, 443 only
- [ ] SSH key-only, tested after the change
- [ ] Agent runs as uid 10000 with no Docker socket
- [ ] Dedicated platform users, custom read-only role, permissions checked
- [ ] Login files owned by 10000, mode 600, never in git
- [ ] Rules deployed and recited after `/reset`
- [ ] Rules, skills and tools pushed to the flowolf repo
