#!/usr/bin/env bash
set -euo pipefail
DEPLOY=/home/evouvali/MailoHLS/deployments/film-gmt-stage2-6ee1390-20261113
test ! -e "$DEPLOY/venv"
python3 -m venv "$DEPLOY/venv"
# Reuse the matching CUDA/Torch stack read-only; dependencies here take priority.
"$DEPLOY/venv/bin/python" -m pip install --no-cache-dir numpy==2.2.6 pandas==2.3.3 python-dateutil==2.9.0.post0 pytz==2025.2 tzdata==2025.2 six==1.17.0
printf '%s\n' /home/evouvali/MailoHLS/.venv/lib/python3.10/site-packages > "$DEPLOY/venv/lib/python3.10/site-packages/davinci_shared_torch.pth"
"$DEPLOY/venv/bin/python" -m pip freeze > "$DEPLOY/receipts/environment_freeze.txt"
