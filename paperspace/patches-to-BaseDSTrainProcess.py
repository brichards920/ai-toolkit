from pathlib import Path
import re
import shutil
import torch

repo_root = Path("/notebooks/ai-toolkit")
target = repo_root / "jobs" / "process" / "BaseSDTrainProcess.py"
backup = target.with_suffix(".py.bak")

print(f"🔧 Repairing {target}")

# 1️⃣ Restore from backup if available
if backup.exists():
    print(f"🗃 Restoring from backup: {backup}")
    shutil.copy(backup, target)
else:
    print("⚠️ No backup found — continuing with current file.")

code = target.read_text()

# 2️⃣ Insert properly indented patch
pattern = r"(\s*)vae\s*=\s*vae\.to\(torch\.device\('cpu'\),\s*dtype=dtype\)"
replacement = r"""\1try:
\1    vae = vae.to(torch.device('cpu'), dtype=dtype)
\1except NotImplementedError as e:
\1    if 'meta tensor' in str(e):
\1        print("[!] Meta VAE detected — using to_empty() safely")
\1        try:
\1            vae = vae.to_empty(device=torch.device('cpu'), dtype=dtype)
\1        except TypeError:
\1            vae = vae.to_empty(device=torch.device('cpu'))
\1    else:
\1        raise
"""

new_code, count = re.subn(pattern, replacement, code)
if count == 0:
    print("⚠️ No match found for vae.to() — check line numbers.")
else:
    target.write_text(new_code)
    print(f"✅ Patch reapplied successfully ({count} replacement).")

print(f"Detected torch version: {torch.__version__}")
