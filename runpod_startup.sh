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

# 6. Claude Code                            
curl -fsSL https://claude.ai/install.sh | bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc