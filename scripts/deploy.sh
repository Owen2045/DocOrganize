#!/bin/bash
# 部署到 Mac mini：傳檔案 + 重啟服務
set -e

REMOTE="owen@100.71.132.50"
REMOTE_DIR="/Users/owen/citefund"

echo "==> Syncing files..."
rsync -av --exclude='.env' --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='.git' --exclude='docs/' \
  ./ "$REMOTE:$REMOTE_DIR/"

echo "==> Restarting app..."
ssh "$REMOTE" "cd $REMOTE_DIR && sudo /opt/homebrew/bin/docker compose restart app 2>/dev/null || echo 'app not running yet'"

echo "==> Done."
