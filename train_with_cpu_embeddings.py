#!/usr/bin/env python3
# Author: Bradley R. Kinnard
# Wrapper script for training on CPU to avoid Blackwell CUDA kernel compatibility issues
# Temporary workaround until PyTorch/transformers have full Blackwell support

import os
import sys
from pathlib import Path

# Force CPU for all torch operations before importing anything
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide GPU from PyTorch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now import and patch config before any modules load
from backend import config
config.cfg['device'] = 'cpu'
config.cfg['torch_device'] = 'cpu'
config.cfg['embedding']['device'] = 'cpu'

print("🔧 Training mode: Full CPU (Blackwell CUDA kernel workaround)")
print("   Embeddings: CPU | PPO Policy: CPU")
print("   Note: Slower but avoids 'no kernel image available' error")

# Now run the actual training
from backend.rl import train_rl
train_rl.main()
