#!/usr/bin/env bash
# Refresh an existing reviewbot-web deployment in place.
#
# Reads .deploy-state.json (written by deploy.sh), then over SSH:
#   - git fetch + reset --hard origin/${REPO_BRANCH} in /opt/app/ai-reviewer
#   - pip install -e .[web] (catches new deps)
#   - rewrite ${APP_ENV_REMOTE_FILE} from the local env file (and the PEM
#     file if GITHUB_PRIVATE_KEY_PATH points somewhere other than the
#     remote pem path)
#   - systemctl restart ${SERVICE_NAME}
#
# Safe to run repeatedly. Does NOT touch the EC2 instance, security
# group, or key pair — only the application bits inside it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_FILE="${SCRIPT_DIR}/.deploy-state.json"

for cmd in aws jq ssh base64; do
  command -v "$cmd" >/dev/null || { echo "missing dependency: $cmd" >&2; exit 1; }
done

if [[ ! -f "$STATE_FILE" ]]; then
  echo "no state file at $STATE_FILE — run deploy.sh first." >&2
  exit 1
fi

REGION="$(jq -r .region "$STATE_FILE")"
INSTANCE_ID="$(jq -r .instance_id "$STATE_FILE")"
KEY_FILE="$(jq -r .key_file "$STATE_FILE")"
PRIVATE_IP="$(jq -r .private_ip "$STATE_FILE")"
SERVICE_NAME="$(jq -r .service_name "$STATE_FILE")"
APP_ENV_LOCAL_FILE="$(jq -r .app_env_local "$STATE_FILE")"
PEM_REMOTE_FILE="$(jq -r .pem_remote "$STATE_FILE")"

REPO_BRANCH="${REPO_BRANCH:-main}"
APP_DIR="/opt/app/ai-reviewer"
APP_ENV_REMOTE_FILE="/etc/reviewbot/${SERVICE_NAME}.env"

if [[ ! -f "$KEY_FILE" ]]; then
  echo "missing SSH key: $KEY_FILE" >&2
  exit 1
fi
if [[ ! -f "$APP_ENV_LOCAL_FILE" ]]; then
  echo "missing env file: $APP_ENV_LOCAL_FILE" >&2
  exit 1
fi

# ---- check instance state + refresh private IP if it moved ------------------
state="$(aws ec2 describe-instances --region "$REGION" --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null || true)"
if [[ "$state" != "running" ]]; then
  echo "instance $INSTANCE_ID is not running (state=$state)" >&2
  exit 1
fi
CURRENT_IP="$(aws ec2 describe-instances --region "$REGION" --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PrivateIpAddress' --output text)"
if [[ "$CURRENT_IP" != "$PRIVATE_IP" ]]; then
  echo "==> private IP changed: $PRIVATE_IP -> $CURRENT_IP; updating state file"
  tmp="$(mktemp)"
  jq --arg ip "$CURRENT_IP" '.private_ip=$ip' "$STATE_FILE" > "$tmp"
  mv "$tmp" "$STATE_FILE"
  PRIVATE_IP="$CURRENT_IP"
fi

# ---- stage env + PEM (mirrors deploy.sh) ------------------------------------
APP_ENV_STAGED_FILE="$(mktemp)"
trap 'rm -f "$APP_ENV_STAGED_FILE"' EXIT
cp "$APP_ENV_LOCAL_FILE" "$APP_ENV_STAGED_FILE"

read_env_value() {
  local key="$1"
  local file="$2"
  local line

  line="$(grep -E "^${key}=" "$file" | tail -n 1 || true)"
  [[ -n "$line" ]] || return 1

  line="${line#*=}"
  line="${line%$'\r'}"
  if [[ "${line:0:1}" == '"' && "${line: -1}" == '"' ]]; then
    line="${line:1:${#line}-2}"
  elif [[ "${line:0:1}" == "'" && "${line: -1}" == "'" ]]; then
    line="${line:1:${#line}-2}"
  fi

  printf '%s' "$line"
}

PEM_LOCAL_FILE="$(read_env_value GITHUB_PRIVATE_KEY_PATH "$APP_ENV_LOCAL_FILE" || true)"
PEM_B64=""
if [[ -n "$PEM_LOCAL_FILE" && "$PEM_LOCAL_FILE" != "$PEM_REMOTE_FILE" ]]; then
  if [[ "$PEM_LOCAL_FILE" == "~/"* ]]; then
    PEM_LOCAL_FILE="${HOME}/${PEM_LOCAL_FILE#~/}"
  elif [[ "$PEM_LOCAL_FILE" != /* ]]; then
    APP_ENV_DIR="$(cd "$(dirname "$APP_ENV_LOCAL_FILE")" && pwd)"
    PEM_LOCAL_FILE="${APP_ENV_DIR}/${PEM_LOCAL_FILE}"
  fi
  if [[ ! -f "$PEM_LOCAL_FILE" ]]; then
    echo "GITHUB_PRIVATE_KEY_PATH does not exist locally: $PEM_LOCAL_FILE" >&2
    exit 1
  fi
  PEM_B64="$(base64 < "$PEM_LOCAL_FILE" | tr -d '\n')"
  awk -v target="$PEM_REMOTE_FILE" '
    /^GITHUB_PRIVATE_KEY_PATH=/ {
      print "GITHUB_PRIVATE_KEY_PATH=" target
      next
    }
    { print }
  ' "$APP_ENV_LOCAL_FILE" > "$APP_ENV_STAGED_FILE"
fi

APP_ENV_B64="$(base64 < "$APP_ENV_STAGED_FILE" | tr -d '\n')"

# ---- remote update ----------------------------------------------------------
SSH_OPTS=(
  -i "$KEY_FILE"
  -o StrictHostKeyChecking=accept-new
  -o UserKnownHostsFile="${SCRIPT_DIR}/.known_hosts"
  -o ConnectTimeout=10
)

REMOTE_SCRIPT=$(cat <<EOF
set -euo pipefail

echo "==> pulling latest ${REPO_BRANCH}"
sudo -u ec2-user git -C "${APP_DIR}" fetch --quiet origin "${REPO_BRANCH}"
sudo -u ec2-user git -C "${APP_DIR}" checkout --quiet "${REPO_BRANCH}"
sudo -u ec2-user git -C "${APP_DIR}" reset --hard "origin/${REPO_BRANCH}"

echo "==> reinstalling deps"
sudo -u ec2-user "${APP_DIR}/.venv/bin/pip" install --quiet --upgrade pip
sudo -u ec2-user "${APP_DIR}/.venv/bin/pip" install --quiet -e "${APP_DIR}[web]"

echo "==> writing ${APP_ENV_REMOTE_FILE}"
base64 -d <<'ENVFILE' | sudo tee "${APP_ENV_REMOTE_FILE}" > /dev/null
${APP_ENV_B64}
ENVFILE
sudo chown root:ec2-user "${APP_ENV_REMOTE_FILE}"
sudo chmod 0640 "${APP_ENV_REMOTE_FILE}"

if [[ -n "${PEM_B64}" ]]; then
  echo "==> writing ${PEM_REMOTE_FILE}"
  base64 -d <<'PEMFILE' | sudo tee "${PEM_REMOTE_FILE}" > /dev/null
${PEM_B64}
PEMFILE
  sudo chown root:ec2-user "${PEM_REMOTE_FILE}"
  sudo chmod 0640 "${PEM_REMOTE_FILE}"
fi

echo "==> restarting ${SERVICE_NAME}"
sudo systemctl restart "${SERVICE_NAME}"
sleep 2
sudo systemctl --no-pager --lines=8 status "${SERVICE_NAME}" || true
EOF
)

echo "==> ssh ec2-user@${PRIVATE_IP}"
ssh "${SSH_OPTS[@]}" "ec2-user@${PRIVATE_IP}" "bash -s" <<<"$REMOTE_SCRIPT"

cat <<EOF

==> done.
    instance:   $INSTANCE_ID
    private ip: $PRIVATE_IP
    app:        http://$PRIVATE_IP:8080

logs: ssh -i "$KEY_FILE" ec2-user@$PRIVATE_IP 'sudo journalctl -u ${SERVICE_NAME} -f'
EOF
