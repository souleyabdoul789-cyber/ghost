#!/usr/bin/env bash
# Compile le moteur natif Ghost pour l'OS et l'architecture de la machine hôte.
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null 2>&1 || { echo "Python 3 est requis pour sélectionner la cible native." >&2; exit 1; }
"$PYTHON" build_native.py
