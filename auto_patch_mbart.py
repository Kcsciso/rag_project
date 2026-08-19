import os, site

# 获取当前环境的 site-packages 根目录
site_packages = site.getsitepackages()[0]
found = False

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

print("🔍 正在环境内进行地毯式搜索，寻找目标文件...")
for root, dirs, files in os.walk(site_packages):
    if "modeling_mbart.py" in files:
        file_path = os.path.join(root, "modeling_mbart.py")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 确认这是我们要找的公式模型文件
        if "UnimerMBartForCausalLM" in content:
            found = True
            if "Auto Patch for cache_position" not in content:
                with open(file_path, "a", encoding="utf-8") as f:
                    f.write("\n" + patch_code + "\n")
                print(f"✅ 致命一击！成功找到并修补文件：\n   {file_path}")
            else:
                print(f"✅ 该文件已被修补过：\n   {file_path}")

if not found:
    print("❌ 未找到目标模型文件，请检查环境！")
