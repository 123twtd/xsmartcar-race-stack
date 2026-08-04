#!/usr/bin/env python3
# Copyright (c) 2026 清影/123twtd
"""公开发布前的仓库结构、凭据、协议和大文件检查。"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_ENTRIES = {
    "run_track_include_fork.py",
    "run_track_include_fork_gai.py",
    "run_track_include_fork_avoid_v3.py",
}

REQUIRED = [
    "LICENSE",
    "COPYRIGHT.md",
    "THIRD_PARTY_NOTICES.md",
    "docs/PROTOCOL.md",
    "upper/run_track_include_fork.py",
    "upper/run_track_include_fork_gai.py",
    "upper/run_track_include_fork_avoid_v3.py",
    "upper/xsmartcar/mymodel/seg_model/inference_model_384_final_int8.rknn",
    "upper/xsmartcar/mymodel/det_model/mbjc_384_int8.rknn",
    "upper/xsmartcar/mymodel/det_model/mbjc_384_int8.rknn.json",
    "upper/xsmartcar/data_npz/race_data.npz",
    "upper/dev_tools/probes/README.md",
    "upper/dev_tools/probes/rknn_det_model_probe.py",
    "upper/dev_tools/probes/rknn_seg_output_probe.py",
    "upper/dev_tools/probes/cap_arvideo_press_test.py",
    "upper/dev_tools/probes/10000 (95).png",
    "upper/dev_tools/ipm_calibration/README.md",
    "upper/dev_tools/ipm_calibration/race_ipm.py",
    "upper/dev_tools/ipm_calibration/assets/calibration_track.png",
    "upper/dev_tools/ipm_calibration/assets/mask_input.png",
    "lower/.cproject",
    "lower/.project",
    "lower/code/protocol.c",
    "lower/user/cpu0_main.c",
]

GENERATED_DIRS = {"Debug", "Release", "__pycache__", ".pytest_cache"}
GENERATED_SUFFIXES = {".pyc", ".o", ".d", ".src", ".map", ".mdf", ".elf", ".hex", ".opt"}
MAX_GITHUB_FILE_BYTES = 100 * 1024 * 1024

NONPUBLIC_TRAINING_PATHS = {
    "training",
    "model_training",
    "upper/training",
    "upper/model_training",
    "upper/datasets",
}

EXPECTED_SHA256 = {
    "upper/xsmartcar/mymodel/seg_model/inference_model_384_final_int8.rknn": "1456432cff629599009566758540b8ed41ac17eeaf66042eaf2b32722f0940ac",
    "upper/xsmartcar/mymodel/det_model/mbjc_384_int8.rknn": "2177bf76cc9b552750c9827c64d788a304bfb2e314dde77c4179d1bb02b29569",
    "upper/xsmartcar/data_npz/race_data.npz": "fe64dd810315d867bac3af392dad220ac9b7b5914a4932ad390567c5f5ce5a53",
}


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8", errors="replace")


def main() -> int:
    errors: list[str] = []

    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append(f"缺少必需文件: {relative}")

    for relative in NONPUBLIC_TRAINING_PATHS:
        if (ROOT / relative).exists():
            errors.append(f"模型训练整理目录不应进入公开发行: {relative}")

    entries = {path.name for path in (ROOT / "upper").glob("run_track*.py")}
    if entries != EXPECTED_ENTRIES:
        errors.append(f"上位机入口集合不正确: {sorted(entries)}")

    models = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "upper/xsmartcar/mymodel").rglob("*.rknn")
    }
    expected_models = {
        "upper/xsmartcar/mymodel/seg_model/inference_model_384_final_int8.rknn",
        "upper/xsmartcar/mymodel/det_model/mbjc_384_int8.rknn",
    }
    if models != expected_models:
        errors.append(f"RKNN 模型集合不正确: {sorted(models)}")

    for relative, expected_hash in EXPECTED_SHA256.items():
        path = ROOT / relative
        if path.is_file():
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                errors.append(f"模型/标定文件校验失败: {relative}")

    for path in ROOT.rglob("*"):
        if ".git" in path.parts:
            continue
        relative = path.relative_to(ROOT).as_posix()
        # .ads/winIDEAWorkspaces/Debug 是调试器配置，不是编译输出目录。
        if path.is_dir() and path.name in GENERATED_DIRS and ".ads" not in path.parts:
            errors.append(f"包含生成目录: {relative}")
        if not path.is_file():
            continue
        if path.suffix.lower() in GENERATED_SUFFIXES:
            errors.append(f"包含构建产物: {relative}")
        if path.stat().st_size >= MAX_GITHUB_FILE_BYTES:
            errors.append(f"文件达到 GitHub 100 MiB 限制: {relative}")
        if path.name.startswith(".env") and path.name != ".env.example":
            errors.append(f"包含环境凭据文件: {relative}")
        if path.name.endswith((".local.yaml", ".secrets.yaml")):
            errors.append(f"包含本地凭据配置: {relative}")

    config = read("upper/xsmartcar/qianfan_api/qianfan_api_config.yaml")
    if "api_key: <apikey>" not in config or "<token>" not in config:
        errors.append("千帆配置未保持 <apikey>/<token> 占位符")

    license_text = read("LICENSE")
    if "GNU GENERAL PUBLIC LICENSE" not in license_text or "Version 3" not in license_text:
        errors.append("LICENSE 不是完整 GPLv3 文本")

    protocol_doc = read("docs/PROTOCOL.md")
    if "L,near_offset,near_yaw,det_offset_cm" not in protocol_doc:
        errors.append("协议文档缺少三字段 L 帧")
    if "C,1,0" not in protocol_doc or "C,0,0" not in protocol_doc:
        errors.append("协议文档缺少软暂停/清锁停车语义")

    ipm_calibration = read("upper/dev_tools/ipm_calibration/race_ipm.py")
    for token in (
        "W_CM, D_CM, NEAR_CM", "BOARD = (-7.5, 7.5, 5.0, 20.0)", "cv2.getPerspectiveTransform",
        "np.savez", "race_data.npz",
    ):
        if token not in ipm_calibration:
            errors.append(f"IPM race 标定工具缺少关键定义: {token}")

    lower_protocol = read("lower/code/protocol.c")
    if '"L,%f,%f,%f"' not in lower_protocol or '"C,%d,%d"' not in lower_protocol:
        errors.append("下位机协议解析与三字段 L/C 契约不一致")

    upper_python = list((ROOT / "upper").rglob("*.py"))
    for path in upper_python:
        text = path.read_text(encoding="utf-8", errors="replace")
        if "Copyright (c) 2026 清影/123twtd" not in text:
            errors.append(f"上位机源码缺少清影/123twtd版权声明: {path.relative_to(ROOT)}")

    for path in upper_python + list((ROOT / "lower/code").rglob("*.[ch]")) + list((ROOT / "lower/user").rglob("*.[ch]")):
        if "Copyright (c) 2026 高志禹" in path.read_text(encoding="utf-8", errors="replace"):
            errors.append(f"人员避让作者代码不应进入本发行: {path.relative_to(ROOT)}")

    if errors:
        print("release check failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    file_count = sum(1 for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.parts)
    size_bytes = sum(path.stat().st_size for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.parts)
    print(f"release check ok: {file_count} files, {size_bytes / 1024 / 1024:.2f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
