from pathlib import Path
import re
import shutil

file_path = Path("/notebooks/ai-toolkit/toolkit/network_mixins.py")
backup = file_path.with_suffix(".py.bak")

print(f"🔧 Preparing to patch {file_path}")

# 1️⃣ Backup or restore original
if backup.exists():
    print(f"🗃 Restoring from backup {backup}")
    shutil.copy(backup, file_path)
else:
    print("💾 Creating backup before patch")
    shutil.copy(file_path, backup)

code = file_path.read_text()

# 2️⃣ Inject correct CUDA-safe merge patch
pattern = r"(^\s*)weight\s*=\s*weight\s*\+\s*multiplier\s*\*\s*\(up_weight\s*@\s*down_weight\)\s*\*\s*scale"

replacement = (
    r"\1# --- Device-safety patch start ---\n"
    r"\1import torch\n"
    r"\1print(f'[Merge-Debug] weight={weight.device}, up={up_weight.device}, down={down_weight.device}')\n"
    r"\1target_device = torch.device('cuda:0')\n"
    r"\1if weight.device != target_device or up_weight.device != target_device or down_weight.device != target_device:\n"
    r"\1    print(f'[Merge-Patch] Moving tensors to {target_device}')\n"
    r"\1    weight = weight.to(target_device)\n"
    r"\1    up_weight = up_weight.to(target_device)\n"
    r"\1    down_weight = down_weight.to(target_device)\n"
    r"\1weight = weight + multiplier * (up_weight @ down_weight) * scale\n"
    r"\1# --- Device-safety patch end ---"
)

new_code, count = re.subn(pattern, replacement, code, flags=re.MULTILINE)
if count == 0:
    print("⚠️ Could not find merge line — manual inspection needed.")
else:
    file_path.write_text(new_code)
    print(f"✅ Patch applied successfully ({count} replacement).")

print("\n🎯 All tensors will now be moved to CUDA before merge.")
print("🧠 Restart the kernel, then re-run:")
print("   python run.py ./config/my_qwen_lora.yml")
print("and look for '[Merge-Patch] Moving tensors to cuda:0' in the log.")
