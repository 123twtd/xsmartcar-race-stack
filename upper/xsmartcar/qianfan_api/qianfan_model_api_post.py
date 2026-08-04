# Copyright (c) 2026 清影/123twtd
"""Qianfan 视觉大模型 API HTTP请求调用 模块
    ————————— 路牌方向分类
调 Qianfan 的 chat completions 接口。

初始化：加载配置文件 qianfan_api_config.yaml，获取 url、api_key、model、prompt。

pipe循环：
1.图像截取模块
2.图像转 data URL 模块
3.调用 Qianfan API 模块
4.解析响应模块
->将direction传给扫线程序，用于选择分叉道路
"""
import base64
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import requests
import yaml


CONFIG_DIR = Path(__file__).resolve().parent
CONFIG_PATH = str(CONFIG_DIR / "qianfan_api_config.yaml")
LOCAL_CONFIG_PATH = str(CONFIG_DIR / "qianfan_api_config.local.yaml")

"""图像截取模块。

只做一件事：根据坐标从原图截取矩形区域。
不依赖 XML、不依赖 API 调用。
"""
def crop_bounding_box(
    image: np.ndarray,
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
    expand_ratio: float = 0.0,
) -> np.ndarray:
    """从原图中截取指定矩形区域。

    参数:
        image: 原图（numpy 数组，cv2 读取的结果）
        xmin, ymin: 左上角坐标
        xmax, ymax: 右下角坐标
        expand_ratio: 四边外扩比例，0.2 = 每边外扩 20%（防运动模糊裁剪不足）

    返回:
        截取后的图像（numpy 数组），可直接传给 image_to_data_url()
    """
    h, w = image.shape[:2]
    if expand_ratio > 0:
        bw = xmax - xmin
        bh = ymax - ymin
        xmin = xmin - int(bw * expand_ratio)
        ymin = ymin - int(bh * expand_ratio)
        xmax = xmax + int(bw * expand_ratio)
        ymax = ymax + int(bh * expand_ratio)
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(w, xmax)
    ymax = min(h, ymax)

    # numpy 切片：image[行,列] = image[y,x]，注意顺序！
    return image[ymin:ymax, xmin:xmax].copy()

def image_to_data_url(image: str | Path | np.ndarray) -> str:
    """接收文件路径或 cv2 图像，返回 data URL。"""
    if isinstance(image, (str, Path)):
        img = cv2.imdecode(np.fromfile(str(image), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {image}")
    elif isinstance(image, np.ndarray):
        img = image
    else:
        raise TypeError(f"不支持的类型: {type(image)}")

    # 限制最长边 640 像素
    h, w = img.shape[:2]
    if max(h, w) > 640:
        scale = 640 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # 编码为 JPEG
    success, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not success:
        raise RuntimeError("图片压缩失败")

    encoded = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"

def load_config(config_path: str | None = None) -> dict:
    """加载千帆配置，不要求把真实凭据写进受版本控制的文件。

    未显式指定路径时，优先读取被 .gitignore 排除的
    qianfan_api_config.local.yaml，不存在时再读取仓库中的占位模板。
    环境变量 QIANFAN_API_KEY 和 QIANFAN_ACCESS_TOKEN 优先级最高。
    """
    if config_path is None:
        local_path = Path(LOCAL_CONFIG_PATH)
        selected_path = local_path if local_path.is_file() else Path(CONFIG_PATH)
    else:
        selected_path = Path(config_path)

    config = yaml.safe_load(selected_path.read_text(encoding="utf-8")) or {}
    if api_key := os.environ.get("QIANFAN_API_KEY"):
        config["api_key"] = api_key
    if token := os.environ.get("QIANFAN_ACCESS_TOKEN"):
        config["token"] = token
    return config

def qianfan_model_api_request(url: str, api_key: str, model: str, prompt: str,image_url:str) -> dict:
    # url = config["url"]
    # api_key = config["api_key"]
    # model = config["model"]
    # prompt = config["prompt"]
    payload = json.dumps({
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "temperature": 0.1,
    })
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    # timeout=(connect_timeout, read_timeout)
    # 之前 read timeout=1.0 异常：requests.request 某些版本对 tuple timeout 解析有 bug
    # 改用 requests.post + 显式 timeout 关键字，确保 read_timeout=8.0
    # 不设 retries，避免网络差时重复发请求浪费等待时间
    response = requests.post(url, headers=headers, data=payload,
                             timeout=(3.0, 8.0))
    response_data = response.json()      # 第 1 次解析：响应体 → dict
    return response_data

def parse_response(data: dict) -> tuple[str, str, str]:
    content_str = data["choices"][0]["message"]["content"]
    result = json.loads(content_str)   # 第 2 次解析：字符串 → dict
    direction = str(result.get("direction", "none")).strip().upper()
    if direction not in {"L", "R"}:
        direction = "none"
    label = result.get("label", "")
    reason = result.get("reason", "")
    return direction, label, reason

# 工具函数：路牌小图 → 千帆 API → 方向字符串
def call_roadsign_api(frame_bgr, box_xyxy, config):
    """裁剪路牌 → 转码 → POST → 解析，返回方向字符串 "L"/"R"/"none"。"""
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    crop = crop_bounding_box(frame_bgr, x1, y1, x2, y2, expand_ratio=0.2)
    data_url = image_to_data_url(crop)                     # 直接用
    try:
        resp = qianfan_model_api_request(
            config["url"], config["api_key"],
            config["model"], config["prompt"], data_url,
        )
        direction, label, reason = parse_response(resp)
        return direction
    except Exception as e:
        print(f"[roadsign-api] failed: {e}")
        return "none"




# def main():


#     if len(sys.argv) < 2:
#         sys.exit("用法: python qianfan_model_api_post.py <image_path>")
#     image_path = sys.argv[1]

#     config = load_config()
#     url = config["url"]
#     api_key = config["api_key"]
#     model = config["model"]
#     prompt = config["prompt"]

#     image_url = image_to_data_url(image_path)
#     response_data = qianfan_model_api_request(url, api_key, model, prompt, image_url)
#     # print(json.dumps(response_data, ensure_ascii=False, indent=2))

#     direction, label, reason = parse_response(response_data)
#     print(f"方向: {direction}")
#     print(f"标签: {label}")
#     print(f"理由: {reason}")

#     return 0


# if __name__ == "__main__":
    # sys.exit(main())
