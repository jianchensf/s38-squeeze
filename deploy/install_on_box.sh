#!/bin/bash
# s38-squeeze box install (hospitality1, as root). Idempotent. Does NOT start the service.
#   git clone https://github.com/jianchensf/s38-squeeze /home/dev/s38-squeeze   (or: cd /home/dev/s38-squeeze && git pull)
#   bash /home/dev/s38-squeeze/deploy/install_on_box.sh
set -e
D=/home/dev/s38-squeeze; cd "$D"
python3 -c "import sys; assert sys.version_info >= (3, 10), sys.version" || { echo "!! python3 >= 3.10 required"; exit 1; }
python3 -c "import aiohttp, numpy, dotenv, pytest" 2>/dev/null && echo "deps OK" || pip3 install -r requirements.txt
mkdir -p state
[ -f .env ] || { cp env.sample .env; echo "!! created $D/.env from sample — optional: fill TELEGRAM_* for FIRE/ARM alerts"; }
chmod 600 .env
python3 -m py_compile squeeze_scanner.py && echo "compile OK"
python3 -m pytest -q -p no:cacheprovider tests && echo "offline tests OK"
cp deploy/s38-squeeze.service /etc/systemd/system/
systemctl daemon-reload
echo "next: python3 squeeze_scanner.py --once --weight 400   (read-only; ~6 min cold start)"
echo "then: systemctl enable --now s38-squeeze && journalctl -u s38-squeeze -f"
