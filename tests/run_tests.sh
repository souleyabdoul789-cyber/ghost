#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

./build.sh
CC_BIN="${CC:-cc}"
if ! command -v "$CC_BIN" >/dev/null 2>&1; then CC_BIN=cc; fi
C_TEST="${TMPDIR:-/tmp}/g-core-test-$$"
trap 'rm -f "$C_TEST"' EXIT

COMMON=(-std=c99 -g -O1 -Wall -Wextra -Werror -pedantic -I c c/tensor.c c/nn.c tests/test_core.c -lm -o "$C_TEST")
if "$CC_BIN" "${COMMON[@]}" -fsanitize=address,undefined -fno-omit-frame-pointer >/dev/null 2>&1; then
    echo "[C] AddressSanitizer + UBSan"
    ASAN_OPTIONS=detect_leaks=1:abort_on_error=1 UBSAN_OPTIONS=halt_on_error=1 "$C_TEST"
else
    echo "[C] Sanitizers non disponibles ; test natif standard"
    "$CC_BIN" "${COMMON[@]}"
    "$C_TEST"
fi

echo "[Python] Tests d'intégration"
PYTHONPATH=python python3 -m unittest discover -s tests -p 'test_*.py' -v
