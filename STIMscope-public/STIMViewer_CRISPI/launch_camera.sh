#!/bin/bash
# Launch the local STIMViewer GUI from this repo, using the active Python env if available
set -e
cd /home/aharonilabjetson2/Desktop/MyScripts/MyUART/STIMViewer_CRISPI

# Prefer conda env python if CONDA_PREFIX set; else fall back to python3 in PATH; then /usr/bin/python3
if [ -n "$CONDA_PREFIX" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
  PY="$CONDA_PREFIX/bin/python"
else
  PY="$(command -v python3 || true)"
  if [ -z "$PY" ]; then PY="/usr/bin/python3"; fi
fi

exec "$PY" main_gui.pyw