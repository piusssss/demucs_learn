import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fix_windows_write import windows_safe_write_and_rename
import dora.utils
dora.utils.write_and_rename = windows_safe_write_and_rename

SEED = 224

if __name__ == "__main__":
    sys.argv = [sys.argv[0]] + [f"seed={SEED}"] + sys.argv[1:]
    print(f"种子: {SEED}")
     
    from demucs.train import main
    main()
