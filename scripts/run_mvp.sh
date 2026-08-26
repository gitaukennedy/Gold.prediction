#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d .venv ]]; then
  python -m venv .venv
fi

source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [[ ! -f .env ]]; then
  cp config.example.env .env
  echo "Created .env from config.example.env; add paper-trading credentials before enabling trading."
fi

python main.py
