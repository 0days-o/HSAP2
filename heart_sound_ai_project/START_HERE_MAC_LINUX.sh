#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if ! command -v python3.11 >/dev/null 2>&1; then
  echo "Python 3.11 is required. Install it from https://www.python.org/downloads/ and rerun this file."
  exit 1
fi
if [ ! -x .venv/bin/python ]; then
  python3.11 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
