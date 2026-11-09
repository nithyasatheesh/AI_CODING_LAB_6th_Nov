# AI Eval Lab — Production Deployment Guide

Two supported ways to run AI Eval Lab for a client programme. Both use only free / open-source software.

| | **Option A — Own server (recommended)** | **Option B — Streamlit Community Cloud** |
|---|---|---|
| Best for | Client programmes, up to ~40 participants per batch | Pilots, small groups up to ~25 |
| Cost | ₹0 (Oracle Cloud Always Free VM) | ₹0 |
| Watch out | Oracle may reclaim an Always Free VM that stays idle for 7 days; keep backups off the server | — |
| Setup time | ~1 hour, once | ~15 minutes |
| Capacity | 2 CPU / 12 GB RAM (Always Free), 3 code runs at once | ~1 GB RAM, 2 code runs at once |
| Storage | PostgreSQL on the server + daily backups | Neon or Supabase free PostgreSQL |
| Your link | `https://your-name.duckdns.org` (or client domain) | `https://….streamlit.app` |
| Always on | Yes | Sleeps when idle; wakes in ~30 s |
| LLM judge | Groq free, OpenAI, or fully local open-source (Ollama) | Groq free or OpenAI |

---

## What is production-ready in this version

- **Sandboxed code execution** — participant code cannot read the app, secrets, other participants' work or
  problem files; cannot start programs, use the network, or load system libraries. Memory (1.5 GB),
  CPU time and file-size limits per run. The LLM API key is held by a separate helper process, so code can't steal it.
- **Permanent storage** in PostgreSQL (`DATABASE_URL`); falls back to local files only if no database is set.
- **Run queue** — at most `MAX_PARALLEL_RUNS` code runs at once; others wait with a "you are in the queue" message.
- **Roles** — participants see only their own work (protected by their own PIN); trainers sign in with
  `TRAINER_PASSWORD` for the dashboard and all settings.
- **Automated tests** (27) on every GitHub push: solution key, sandbox attacks, storage, end-to-end app flow.
- **Hardened container** (Option A): non-root user, read-only filesystem, no Linux capabilities, HTTPS with
  security headers, daily backups kept 14 days, log rotation, health checks, auto-restart.
- **Client branding** via `APP_NAME` / `APP_TAGLINE`; Streamlit's developer menu is hidden.

---

## Before either option

1. **Make the GitHub repository private** (GitHub → repo → Settings → General → Danger Zone → Change visibility).
   The repo contains the problem rubrics and the reference solutions (`tests/solutions/`). Streamlit Cloud and
   your server can still deploy from a private repo.
2. Place the files so the repo looks like this (the workflow must be at the repo root):
   ```
   AI_CODING_LAB/
   ├── .github/workflows/tests.yml
   └── ai-eval-lab/   (app.py, evaluator/, problems/, data/, deploy/, tests/, Dockerfile …)
   ```
3. Push. On GitHub, open the **Actions** tab — the "AI Eval Lab tests" run should turn green (≈3 min).

---

## Option A — Own server on Oracle Cloud Always Free (recommended)

### A1. Create the free server (≈20 min, one time)
1. Sign up at **cloud.oracle.com** (card needed for verification only; Always Free resources are not charged).
2. **Compute → Instances → Create instance**
   - Image: **Canonical Ubuntu 24.04**
   - Shape: **Ampere VM.Standard.A1.Flex** → 2 OCPUs, 12 GB memory (the current Always Free limit)
   - Add your SSH key; create.
3. **Networking → Virtual cloud network → your subnet → Default security list → Add ingress rules:**
   source `0.0.0.0/0`, TCP, destination ports **80** and **443**.
4. Note the instance's **public IP**.

### A2. Free domain with HTTPS (≈5 min)
1. Go to **duckdns.org**, sign in, create a subdomain (e.g. `xlri-evallab`) and set it to the server IP.
   (Or ask the client's IT to point a subdomain such as `evallab.clientbank.com` to the IP.)

### A3. Install (≈15 min)
```bash
ssh ubuntu@<server-ip>
git clone https://github.com/<you>/AI_CODING_LAB.git      # private repo: use a GitHub token or deploy key
cd AI_CODING_LAB/ai-eval-lab/deploy
./install.sh
```
The installer installs Docker, opens the firewall, generates strong passwords, asks for your domain and
(optional) Groq key, builds and starts everything, and prints the trainer password and your HTTPS link.

Edit settings later in `deploy/.env` (branding, LLM keys, `MAX_PARALLEL_RUNS`) then run `docker compose up -d`.

### A4. Fully self-hosted open-source LLM (optional, no API keys at all)
In `deploy/.env` uncomment:
```
OLLAMA_BASE_URL=http://ollama:11434/v1
OLLAMA_MODEL=qwen2.5-coder:7b
```
then:
```bash
docker compose --profile llm up -d ollama
docker compose exec ollama ollama pull qwen2.5-coder:7b
docker compose up -d app
```
The free server has only 12 GB RAM, so use a small model (`qwen2.5-coder:3b`) and set `MAX_PARALLEL_RUNS=2`.
A judgement takes ~30–90 s per task on CPU. Groq's free tier is far faster; use Ollama only when data must not leave the server.

### A5. Day-to-day commands (run inside `deploy/`)
| Task | Command |
|---|---|
| Status | `docker compose ps` |
| App logs | `docker compose logs -f app` |
| Update to latest GitHub code | `./update.sh` |
| Restart | `docker compose restart app` |
| Backups | automatic daily in `deploy/backups/` (kept 14 days) |
| Manual backup now | `docker compose exec db pg_dump -U evallab evallab \| gzip > backups/manual.sql.gz` |
| Restore a backup | `./restore.sh backups/<file>.sql.gz` |
| Download backups to laptop | `scp ubuntu@<ip>:AI_CODING_LAB/ai-eval-lab/deploy/backups/* .` |

---

## Option B — Streamlit Community Cloud + free PostgreSQL

1. **Free database:** create a project at **neon.tech** (recommended — never pauses permanently) and copy the
   connection string. Supabase also works: use the **Session pooler** string; note free Supabase projects pause
   after 7 days without use and must be resumed in their dashboard before a session.
2. **share.streamlit.io → Create app** → your repo, branch `main`, file `ai-eval-lab/app.py`.
3. **Advanced settings → Python 3.12**, and paste Secrets (see `.streamlit/secrets.toml.example`):
   ```
   TRAINER_PASSWORD = "..."
   DATABASE_URL = "postgresql://...?sslmode=require"
   GROQ_API_KEY = "gsk_..."
   APP_NAME = "Client Programme — AI Lab"
   MAX_PARALLEL_RUNS = "2"
   ```
4. Deploy. Sign in as Trainer — the sidebar must say **"Storage: PostgreSQL database"**.

---

## Session-day runbook

**The day before**
- [ ] GitHub Actions is green for the latest commit; no code pushes until the session ends.
- [ ] Open the app, sign in as Trainer: storage says *PostgreSQL*, LLM judge is set, rules are as wanted
      (copy-paste lock, time-up lock, leaderboard).
- [ ] Do one dry run as a participant (use the solution key) and check it appears on the dashboard. Then reset that test participant.
- [ ] Option B only: open the app 10 minutes before start so it is awake.

**At the start**, tell participants:
> Open the link → **Participant** → enter your full name and email → **make up a PIN and write it down** →
> press Enter → **Start assessment**. If you refresh, use the same email and PIN to continue.

**During** — keep the Trainer Dashboard open (auto-refresh 30 s). Forgotten PIN → *Problems & participants →
Reset selected participant*.

**After**
- [ ] Download *Participant summary*, *Criterion-level scores* and *Submission log* CSVs.
- [ ] Option A: copy the latest backup off the server.

---

## Capacity guide

| Participants at once | Option | `MAX_PARALLEL_RUNS` |
|---|---|---|
| up to 25 | B (Streamlit Cloud) | 2 |
| up to 40 | A (Always Free: 2 CPU / 12 GB) | 3 |
| 40–80 | A on a paid 4 CPU / 24 GB VM (≈ ₹2,500–4,000/month, only while running) | 6 |
| 80+ | A on a larger VM, or split into two batches | 8–12 |

Most participants spend most of their time typing; code runs take 1–5 s each, so the queue rarely exceeds a few seconds.

---

## Security notes (what to tell a client's IT team)
- HTTPS everywhere; HSTS and security headers via Caddy.
- Participant code is executed in a restricted subprocess (Python audit-hook sandbox + OS resource limits)
  inside a non-root, read-only, capability-free container. No network or process creation from participant code.
- Secrets live only in `deploy/.env` (mode 600) or Streamlit's secret store — never in Git.
- Participant work is PIN-protected per email; trainer access is password-protected.
- All results are in PostgreSQL with daily backups; nothing is sent to third parties except prompts to the
  chosen LLM provider (none at all with the self-hosted Ollama option).
- For very high-stakes exams, add per-participant one-time access codes and invigilation (roadmap).

## Troubleshooting
| Symptom | Fix |
|---|---|
| Browser shows "connection refused" / no HTTPS | Ports 80/443 not open in Oracle security list, or DuckDNS not pointing to the IP. `docker compose logs caddy` |
| Sidebar says "temporary files" | `DATABASE_URL` missing/wrong. Option A: `docker compose ps db`. Option B: check the Secrets string |
| "Server busy — you are in the queue" for long | Raise `MAX_PARALLEL_RUNS` (Option A) or move to Option A |
| "Your code used more than 1536 MB" | Participant loaded something huge; ask them to sample data |
| LLM feedback says "LLM judge unavailable" | Check the API key / Ollama container; scores fall back to the rule-based estimate |
| Start button greyed out | Name, Email and a 4+ character PIN must be filled — press Enter after each |
