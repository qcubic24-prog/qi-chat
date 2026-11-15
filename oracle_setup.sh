#!/bin/bash
# Qi Always On: sets up an Oracle Cloud "Always Free" server (Ubuntu, Ampere A1, 2 cores, 12 GB) to run the
# Qi Pro brain all the time, with web search, and list itself on https://qcubic24-prog.github.io/qi-chat/
#
# Paste this into the instance's "Initialization script" (cloud-init) box when you create it:
#   #!/bin/bash
#   export QI_MODEL_ID=<Drive id of Qi Trials/qi-pro-brain/qi-pro-brain.gguf>
#   curl -sL https://qcubic24-prog.github.io/qi-chat/oracle_setup.sh | bash
# Setup takes about 30 minutes. Progress is written to /var/log/qi-setup.log. Safe to run again.
set -u
exec >> /var/log/qi-setup.log 2>&1
echo "=== Qi setup started $(date)"
PAGE=https://qcubic24-prog.github.io/qi-chat
Q=/opt/qi
: "${QI_MODEL_ID:?set QI_MODEL_ID to the Drive id of the Qi Pro brain}"

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y build-essential cmake git curl python3-venv python3-pip ca-certificates

id -u qi >/dev/null 2>&1 || useradd -r -m -d $Q -s /usr/sbin/nologin qi
mkdir -p $Q /var/lib/qi/chats
cd $Q

# Python parts: chat server, web search, Drive download
[ -x $Q/venv/bin/python ] || python3 -m venv $Q/venv
$Q/venv/bin/pip install -q -U fastapi uvicorn httpx ddgs gdown

# The brain engine (llama.cpp), built for this processor
if [ ! -x $Q/llama.cpp/build/bin/llama-server ]; then
  rm -rf $Q/llama.cpp
  git clone --depth 1 --branch b12307 https://github.com/ggml-org/llama.cpp $Q/llama.cpp
  cmake -S $Q/llama.cpp -B $Q/llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF
  cmake --build $Q/llama.cpp/build --target llama-server -j "$(nproc)"
fi

# The free Cloudflare internet link
ARCH=$(uname -m); [ "$ARCH" = aarch64 ] && CF=arm64 || CF=amd64
[ -x $Q/cloudflared ] || { curl -sL -o $Q/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-$CF && chmod +x $Q/cloudflared; }

# The Qi Pro brain from the "Qi Trials" Drive folder
if [ ! -s $Q/qi-pro-brain.gguf ] || [ "$(stat -c %s $Q/qi-pro-brain.gguf)" -lt 5000000000 ]; then
  $Q/venv/bin/gdown --id "$QI_MODEL_ID" -O $Q/qi-pro-brain.gguf || echo "!!! brain download failed; run this script again later"
fi

# The Qi chat server parts (fetched again on every start, so fixes arrive by themselves)
for f in qi_server.py qi_register.py qi_link.py; do curl -sL -o $Q/$f "$PAGE/$f?t=$(date +%s)"; done
[ -s $Q/key ] || echo "qi-$(head -c 8 /dev/urandom | od -An -tx1 | tr -d ' \n')" > $Q/key
cat > $Q/qi.env <<EOF
QI_KEY=$(cat $Q/key)
QI_NAME=Qi Always On
QI_SLUG=qi-oracle
QI_ABOUT=Oracle Cloud (always on, 2 processor cores, no graphics card, so it writes a few words per second)
QI_SLOTS=2
QI_STORE=/var/lib/qi/chats
QI_KEEP_DAYS=3
QI_CLOUDFLARED=$Q/cloudflared
EOF
chown -R qi:qi $Q /var/lib/qi

cat > /etc/systemd/system/qi-brain.service <<EOF
[Unit]
Description=Qi Pro brain
After=network-online.target
[Service]
User=qi
ExecStart=$Q/llama.cpp/build/bin/llama-server -m $Q/qi-pro-brain.gguf --host 127.0.0.1 --port 8088 -c 16384 -np 2 -t $(nproc) --jinja --chat-template-kwargs '{"enable_thinking":false}'
Restart=always
RestartSec=10
[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/qi-chat.service <<EOF
[Unit]
Description=Qi chat server
After=qi-brain.service network-online.target
[Service]
User=qi
WorkingDirectory=$Q
EnvironmentFile=$Q/qi.env
ExecStartPre=-/usr/bin/curl -sfL -o $Q/qi_server.py $PAGE/qi_server.py
ExecStart=$Q/venv/bin/python -m uvicorn --app-dir $Q qi_server:app --host 127.0.0.1 --port 8090
Restart=always
RestartSec=10
[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/qi-link.service <<EOF
[Unit]
Description=Qi internet link and website listing
After=qi-chat.service network-online.target
[Service]
User=qi
WorkingDirectory=$Q
EnvironmentFile=$Q/qi.env
ExecStartPre=-/usr/bin/curl -sfL -o $Q/qi_register.py $PAGE/qi_register.py
ExecStartPre=-/usr/bin/curl -sfL -o $Q/qi_link.py $PAGE/qi_link.py
ExecStart=$Q/venv/bin/python $Q/qi_link.py
Restart=always
RestartSec=15
[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now qi-brain qi-chat qi-link
systemctl restart qi-brain qi-chat qi-link
echo "=== Qi setup finished $(date). Qi Always On will appear on $PAGE in a minute or two."
