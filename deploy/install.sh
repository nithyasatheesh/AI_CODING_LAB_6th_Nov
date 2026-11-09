#!/usr/bin/env bash
# One-command install on a fresh Ubuntu 22.04/24.04 server (e.g. Oracle Cloud Always Free).
# Usage:  cd ai-eval-lab/deploy && ./install.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "==> 1/5 Installing Docker (if missing)"
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER" || true
fi
DOCKER="docker"; docker info >/dev/null 2>&1 || DOCKER="sudo docker"

echo "==> 2/5 Opening ports 80 and 443 in the server firewall"
if command -v ufw >/dev/null 2>&1 && sudo ufw status | grep -q active; then
  sudo ufw allow 80/tcp && sudo ufw allow 443/tcp
fi
# Oracle Cloud Ubuntu images block these with iptables by default
if sudo iptables -L INPUT -n 2>/dev/null | grep -q "REJECT"; then
  sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
  sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
  sudo apt-get install -y iptables-persistent >/dev/null 2>&1 || true
  sudo netfilter-persistent save >/dev/null 2>&1 || true
fi

echo "==> 3/5 Preparing settings (.env)"
if [ ! -f .env ]; then
  cp .env.example .env
  gen() { tr -dc 'A-Za-z0-9' </dev/urandom | head -c 24; }
  sed -i "s/^POSTGRES_PASSWORD=CHANGE_ME/POSTGRES_PASSWORD=$(gen)/" .env
  TP=$(gen | head -c 12); sed -i "s/^TRAINER_PASSWORD=CHANGE_ME/TRAINER_PASSWORD=$TP/" .env
  echo "    Trainer password generated: $TP   (also stored in deploy/.env)"
  read -rp "    Your domain (e.g. evallab.duckdns.org): " DOM
  sed -i "s/^DOMAIN=.*/DOMAIN=$DOM/" .env
  read -rp "    Groq API key (Enter to skip): " GK
  [ -n "$GK" ] && sed -i "s/^GROQ_API_KEY=.*/GROQ_API_KEY=$GK/" .env
fi
chmod 600 .env
mkdir -p backups

echo "==> 4/5 Building and starting"
$DOCKER compose up -d --build
if grep -q "^OLLAMA_BASE_URL=" .env; then
  $DOCKER compose --profile llm up -d ollama
  MODEL=$(grep ^OLLAMA_MODEL= .env | cut -d= -f2); MODEL=${MODEL:-qwen2.5-coder:3b}
  echo "    Downloading LLM $MODEL (several GB, one-time)…"
  $DOCKER compose exec -T ollama ollama pull "$MODEL"
fi

echo "==> 5/5 Health check"
for i in $(seq 1 30); do
  if $DOCKER compose exec -T app python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')" 2>/dev/null; then
    echo; echo "✅ AI Eval Lab is running:  https://$(grep ^DOMAIN= .env | cut -d= -f2)"; exit 0
  fi
  sleep 3
done
echo "⚠️  App not healthy yet — check:  $DOCKER compose logs app"; exit 1
