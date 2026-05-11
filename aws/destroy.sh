#!/usr/bin/env bash
# Tear down the EC2 host provisioned by deploy.sh using the local state file.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_FILE="${SCRIPT_DIR}/.deploy-state.json"

for cmd in aws jq; do
  command -v "$cmd" >/dev/null || { echo "missing dependency: $cmd" >&2; exit 1; }
done

if [[ ! -f "$STATE_FILE" ]]; then
  echo "no state file at $STATE_FILE — nothing to destroy." >&2
  exit 1
fi

REGION="$(jq -r .region "$STATE_FILE")"
INSTANCE_ID="$(jq -r .instance_id "$STATE_FILE")"
SG_ID="$(jq -r .sg_id "$STATE_FILE")"
KEY_NAME="$(jq -r .key_name "$STATE_FILE")"
KEY_FILE="$(jq -r .key_file "$STATE_FILE")"
SG_OURS="$(jq -r '.sg_created_by_us // false' "$STATE_FILE")"
KEY_OURS="$(jq -r '.key_created_by_us // true' "$STATE_FILE")"

echo "about to destroy:"
jq . "$STATE_FILE"
read -r -p "proceed? [y/N] " reply
[[ "$reply" =~ ^[Yy]$ ]] || { echo "aborted."; exit 1; }

echo "==> terminating $INSTANCE_ID"
if aws ec2 describe-instances --region "$REGION" --instance-ids "$INSTANCE_ID" \
     --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null \
     | grep -qv terminated; then
  aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE_ID" >/dev/null
  aws ec2 wait instance-terminated --region "$REGION" --instance-ids "$INSTANCE_ID"
else
  echo "    already gone"
fi

if [[ "$SG_OURS" == "true" ]]; then
  echo "==> deleting security group $SG_ID"
  aws ec2 delete-security-group --region "$REGION" --group-id "$SG_ID" 2>/dev/null \
    || echo "    already gone or still referenced"
else
  echo "==> leaving security group $SG_ID (not created by this stack)"
fi

if [[ "$KEY_OURS" == "true" ]]; then
  echo "==> deleting key pair $KEY_NAME"
  aws ec2 delete-key-pair --region "$REGION" --key-name "$KEY_NAME" 2>/dev/null \
    || echo "    already gone"
else
  echo "==> leaving key pair $KEY_NAME (not created by this stack)"
fi

if [[ -f "$KEY_FILE" ]]; then
  read -r -p "delete local private key $KEY_FILE? [y/N] " reply
  if [[ "$reply" =~ ^[Yy]$ ]]; then
    rm -f "$KEY_FILE"
    echo "    removed"
  fi
fi

rm -f "$STATE_FILE"
echo "==> done."
