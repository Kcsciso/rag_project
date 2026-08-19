import os, sys

try:
    import unimernet
    file_path = os.path.join(os.path.dirname(unimernet.__file__), "models", "mbart", "modeling_mbart.py")
except ImportError:
    import site
    file_path = os.path.join(site.getsitepackages()[0], "unimernet", "models", "mbart", "modeling_mbart.py")

if not os.path.exists(file_path):
    print(f"❌ 找不到文件: {file_path}")
    sys.exit(1)

patch_code = """
# --- Auto Patch for cache_position ---
try:
    if not hasattr(UnimerMBartForCausalLM, "_is_patched"):
        _orig_forward = UnimerMBartForCausalLM.forward
        def _patched_forward(self, *args, **kwargs):
            kwargs.pop("cache_position", None)
            kwargs.pop("num_logits_to_keep", None)
            return _orig_forward(self, *args, **kwargs)
        UnimerMBartForCausalLM.forward = _patched_forward
        UnimerMBartForCausalLM._is_patched = True
except Exception:
    pass
"""

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

if "Auto Patch for cache_position" not in content:
    with open(file_path, "a", encoding="utf-8") as f:
        f.write("\n" + patch_code + "\n")
    print("✅ 降维打击成功：已将补丁永远焊死在 unimernet 底层源码中！")
else:
    print("✅ 源码补丁已存在。")
