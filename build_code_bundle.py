import os
from pathlib import Path
from datetime import datetime

# ===================== 配置区 =====================
# 输出的 bundle 文件名
OUTPUT_FILE = "PROJECT_CODE_BUNDLE.txt"

# 需要包含的文件后缀
INCLUDE_SUFFIX = {".py", ".md", ".txt", ".yaml", ".toml", ".cfg"}

# 需要排除的目录
EXCLUDE_DIRS = {
    ".git", "__pycache__", "venv", ".venv", "outputs",
    ".vscode", ".idea", "__MACOSX", "build", "dist",
    ".ipynb_checkpoints"
}

# 需要排除的具体文件
EXCLUDE_FILES = {
    OUTPUT_FILE, "build_code_bundle.py",
    "test_global_skip.py",
    "test_H_glob.py",
    "test_kip_gate.py",
    "test_norm_envs.py",
    "test_psi_top_revocation.py",
    "test_right_left_envs.py"
}
# ==================================================

def should_include(file_path: Path) -> bool:
    """判断文件是否需要加入打包"""
    if file_path.name in EXCLUDE_FILES:
        return False
    if file_path.suffix not in INCLUDE_SUFFIX:
        return False
    if file_path.name.startswith("."):
        return False
    return True

def main():
    root = Path.cwd()
    output_lines = []

    # 头部信息
    output_lines.append("=" * 72)
    output_lines.append(f"项目代码全量快照")
    output_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append(f"项目根目录: {root.resolve()}")
    output_lines.append("=" * 72)
    output_lines.append("")

    file_count = 0

    # 递归遍历目录
    for dirpath, dirnames, filenames in os.walk(root):
        # 原地过滤目录，os.walk 不会进入被排除的目录
        dirnames[:] = [
            d for d in dirnames
            if d not in EXCLUDE_DIRS and not d.startswith(".")
        ]

        current_dir = Path(dirpath)
        for filename in filenames:
            file_path = current_dir / filename
            if not should_include(file_path):
                continue

            rel_path = file_path.relative_to(root)
            file_count += 1
            print(f"  打包: {rel_path}")

            # 写入文件分隔块
            output_lines.append("=" * 72)
            output_lines.append(f"文件路径: {rel_path}")
            output_lines.append("=" * 72)

            # 读取文件内容
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                output_lines.append(content)
            except UnicodeDecodeError:
                output_lines.append("[文件编码非 UTF-8，已跳过读取]")
            output_lines.append("")
            output_lines.append("")

    # 写入最终文件
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"\n✅ 生成完成，共打包 {file_count} 个文件")
    print(f"   输出文件: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()