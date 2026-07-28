#!/bin/sh
set -eu
STATE_DIR="${GLM_STORAGE_DIR:-/var/data}"
mkdir -p "$STATE_DIR" 2>/dev/null || true
chmod u+rwx "$STATE_DIR" 2>/dev/null || true
exec "$@"
