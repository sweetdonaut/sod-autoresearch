#!/bin/bash
set -euo pipefail                                                                                  
                                                                                                                                                           
# 1. venv     
mkdir -p /root/.config/Ultralytics                                                                 
export YOLO_CONFIG_DIR=/root/.config
uv venv /root/sod-env --python python3.12                                                               
source /root/sod-env/bin/activate                                                                  
uv pip install ultralytics==8.4.36
uv pip install torch==2.10.0 torchvision==0.25.0
uv cache clean
yolo settings datasets_dir=/root/datasets

# 2. Pre-flight
if [ ! -d /workspace/datasets/VisDrone/images/train ]; then
    echo "ERROR: /workspace/datasets/VisDrone incomplete (missing images/train)."
    echo "Download it once via: yolo settings datasets_dir=/workspace/datasets && \\"
    echo "  python -c \"from ultralytics.data.utils import check_det_dataset; check_det_dataset('VisDrone.yaml')\""
    exit 1
fi

# 3. Copy VisDrone to local (~2.3 GB, ~1-2 min)
if [ ! -d /root/datasets/VisDrone ]; then
    echo "Copying VisDrone from volume to local disk..."
    mkdir -p /root/datasets
    rsync -a --info=progress2 /workspace/datasets/VisDrone /root/datasets/
fi

# 4. Persist env for next shells
cat >> ~/.bashrc <<'EOF'
export YOLO_CONFIG_DIR=/root/.config
export VISDRONE_ROOT=/root/datasets/VisDrone
source /root/sod-env/bin/activate
cd /workspace/sod-autoresearch
EOF

# 5. Apply for current run too
export VISDRONE_ROOT=/root/datasets/VisDrone
cd /workspace/sod-autoresearch

# 6. Claude Code install (writes to ephemeral ~/.claude, replaced in step 7)
curl -fsSL https://claude.ai/install.sh | bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# 7. Persist Claude Code state to /workspace (survives pod restart)
#    ~/.claude is on the ephemeral overlay FS — would lose all sessions/memory
#    on pod terminate. Symlink it to /workspace so state survives.
#    MUST run after install so install's writes go to ephemeral dir we discard,
#    not overwrite the preserved state in /workspace/.claude.
CLAUDE_PERSIST=/workspace/.claude
if [ ! -e "$CLAUDE_PERSIST" ]; then
    if [ -d /workspace/claude-home-backup ]; then
        echo "Restoring Claude state from /workspace/claude-home-backup..."
        mv /workspace/claude-home-backup "$CLAUDE_PERSIST"
    elif [ -d "$HOME/.claude" ]; then
        # First-time setup ever: persist whatever install just created
        echo "First-time setup: migrating fresh ~/.claude → $CLAUDE_PERSIST"
        mv "$HOME/.claude" "$CLAUDE_PERSIST"
    else
        mkdir -p "$CLAUDE_PERSIST"
    fi
fi
# Replace ~/.claude (whatever install made) with symlink to persistent
if [ -e "$HOME/.claude" ] && [ ! -L "$HOME/.claude" ]; then
    rm -rf "$HOME/.claude"
fi
[ -L "$HOME/.claude" ] || ln -s "$CLAUDE_PERSIST" "$HOME/.claude"
echo "~/.claude -> $CLAUDE_PERSIST (persistent)"