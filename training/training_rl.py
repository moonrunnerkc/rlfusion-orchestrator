# Author: Bradley R. Kinnard
# training/training_rl.py - DEPRECATED, use backend/rl/train_rl.py instead
# This file is kept for backward compatibility with run_training.sh

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.rl.train_rl import main

if __name__ == "__main__":
    print("⚠️  training/training_rl.py is deprecated")
    print("   Use: python -m backend.rl.train_rl instead")
    print()
    main()
