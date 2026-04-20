#!/usr/bin/env bash
# Launcher para o Chassi Reader — ativa o venv e inicia o app
set -e
cd "$(dirname "$(readlink -f "$0")")"
source venv/bin/activate
exec python3 main.py
