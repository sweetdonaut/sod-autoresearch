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
if [ ! -d /workspace/datasets/coco/images/train2017 ]; then                                                 
    echo "ERROR: /workspace/datasets/coco incomplete. Run pod_fetch_coco.sh first."
    exit 1                                                                                         
fi                                                     
                                                                                                    
# 3. Copy COCO to local                     
if [ ! -d /root/datasets/coco ]; then                                                              
    echo "Copying COCO from volume to local disk (~5 min)..."
    mkdir -p /root/datasets                                                                        
    rsync -a --info=progress2 /workspace/datasets/coco /root/datasets/
fi                                                                                                 
                                                                                                    
# 4. Persist env for next shells            
cat >> ~/.bashrc <<'EOF'
export YOLO_CONFIG_DIR=/root/.config                                                
export COCO_ROOT=/root/datasets/coco    
source /root/sod-env/bin/activate                                                                  
cd /workspace/sod-autoresearch                                                                     
EOF

# 5. Apply for current run too                                                                     
export COCO_ROOT=/root/datasets/coco
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