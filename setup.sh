#!/usr/bin/env bash
set -euo pipefail

require_cmd() {
  command -v "$1" >/dev/null 2>&1
}

log() { printf "\033[1;34m[setup]\033[0m %s\n" "$*"; }
ok()  { printf "\033[1;32m[ok]\033[0m %s\n" "$*"; }
warn(){ printf "\033[1;33m[warn]\033[0m %s\n" "$*"; }
err() { printf "\033[1;31m[err]\033[0m %s\n" "$*" >&2; }

USER_NAME="${SUDO_USER:-${USER}}"

# 1) Ensure 'docker' group exists
if getent group docker >/dev/null 2>&1; then
  ok "Group 'docker' already exists."
else
  log "Creating group 'docker'..."
  sudo groupadd docker
  ok "Created group 'docker'."
fi

# 2) Ensure current user is in 'docker' group
if id -nG "${USER_NAME}" | tr ' ' '\n' | grep -qx docker; then
  ok "User '${USER_NAME}' already in group 'docker'."
  GROUP_CHANGED=0
else
  log "Adding '${USER_NAME}' to group 'docker'..."
  sudo usermod -aG docker "${USER_NAME}"
  ok "Added '${USER_NAME}' to 'docker'."
  GROUP_CHANGED=1
fi

# 3) Enable & start the Docker daemon
if systemctl is-enabled --quiet docker 2>/dev/null; then
  ok "docker.service already enabled."
else
  log "Enabling docker.service..."
  sudo systemctl enable docker
  ok "Enabled docker.service."
fi

if systemctl is-active --quiet docker 2>/dev/null; then
  ok "docker.service already running."
else
  log "Starting docker.service..."
  sudo systemctl start docker
  ok "Started docker.service."
fi

ok "Docker post-install setup looks good for Dev Containers."
