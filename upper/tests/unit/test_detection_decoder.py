# Copyright (c) 2026 清影/123twtd
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

# 把仓库根目录加入 sys.path，保证直接运行本脚本时也能 import xsmartcar
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from xsmartcar.myinference.det_infer.detector import (
    LetterboxMeta,
    detections_from_arrays,
    per_class_nms,
    unletterbox_boxes,
    decode_outputs_fast,
)


# ── 参考实现：旧版未优化的逐分支解码，仅用于和 decode_outputs_fast 对比 ──

def _ref_softmax(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.maximum(np.sum(exp_x, axis=axis, keepdims=True), 1e-6)


def _ref_dfl(position: np.ndarray) -> np.ndarray:
    n, c, h, w = position.shape
    bins = c // 4
    reshaped = position.reshape(n, 4, bins, h, w)
    prob = _ref_softmax(reshaped, axis=2)
    grid = np.arange(bins, dtype=np.float32).reshape(1, 1, bins, 1, 1)
    return np.sum(prob * grid, axis=2)


def _ref_decode_branch(position, class_scores, obj_scores, input_hw):
    input_h, input_w = input_hw
    _, _, grid_h, grid_w = position.shape
    col, row = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
    grid = np.stack([col, row], axis=0).astype(np.float32).reshape(1, 2, grid_h, grid_w)
    stride = np.array([input_w / grid_w, input_h / grid_h], dtype=np.float32).reshape(1, 2, 1, 1)

    dist = _ref_dfl(position)
    top_left = (grid + 0.5 - dist[:, 0:2]) * stride
    bottom_right = (grid + 0.5 + dist[:, 2:4]) * stride
    boxes_xyxy = np.concatenate([top_left, bottom_right], axis=1)
    boxes_xyxy = boxes_xyxy.transpose(0, 2, 3, 1).reshape(-1, 4)

    cls = class_scores.transpose(0, 2, 3, 1).reshape(-1, class_scores.shape[1])
    if obj_scores is not None:
        obj = obj_scores.transpose(0, 2, 3, 1).reshape(-1, 1)
        final_scores = cls * obj
    else:
        final_scores = cls
    return boxes_xyxy, final_scores


def reference_decode(outputs, *, conf_thr, iou_thr, class_names, meta, input_hw):
    per_branch = len(outputs) // 3
    boxes_parts = []
    score_parts = []
    for branch in range(3):
        base = branch * per_branch
        boxes, scores = _ref_decode_branch(
            outputs[base],
            outputs[base + 1],
            outputs[base + 2] if per_branch == 3 else None,
            input_hw,
        )
        boxes_parts.append(boxes)
        score_parts.append(scores)
    boxes = np.concatenate(boxes_parts)
    scores = np.concatenate(score_parts)
    class_ids = np.argmax(scores, axis=1)
    best_scores = scores[np.arange(scores.shape[0]), class_ids]
    selected = np.flatnonzero(best_scores >= conf_thr)
    boxes = unletterbox_boxes(boxes[selected], meta)
    class_ids = class_ids[selected]
    best_scores = best_scores[selected]
    keep = per_class_nms(boxes, class_ids, best_scores, iou_thr)
    return detections_from_arrays(boxes[keep], class_ids[keep], best_scores[keep], class_names)


class DetectionDecoderTests(unittest.TestCase):
    def make_outputs(self, with_objectness: bool):
        rng = np.random.default_rng(42)
        outputs = []
        for height, width in ((4, 4), (2, 2), (1, 1)):
            outputs.append(rng.normal(size=(1, 16, height, width)).astype(np.float32))
            outputs.append(rng.random(size=(1, 3, height, width), dtype=np.float32))
            if with_objectness:
                outputs.append(rng.random(size=(1, 1, height, width), dtype=np.float32))
        return outputs

    def check_format(self, with_objectness: bool):
        outputs = self.make_outputs(with_objectness)
        names = ["gold", "car", "human"]
        meta = LetterboxMeta(1.0, 0, 0, 32, 32, 32, 32)
        kwargs = dict(conf_thr=0.35, iou_thr=0.45, class_names=names, input_hw=(32, 32))
        expected = reference_decode(outputs, meta=meta, **kwargs)
        actual = decode_outputs_fast(outputs, letterbox_meta=meta, **kwargs)
        self.assertEqual(len(actual), len(expected))
        for got, want in zip(actual, expected):
            self.assertEqual(got.class_id, want.class_id)
            self.assertAlmostEqual(got.score, want.score, places=6)
            np.testing.assert_allclose(got.box_xyxy, want.box_xyxy, rtol=1e-5, atol=1e-5)

    def test_six_output_head_matches_reference(self):
        self.check_format(False)

    def test_nine_output_head_matches_reference(self):
        self.check_format(True)

    def test_rejects_unknown_output_layout(self):
        with self.assertRaises(RuntimeError):
            decode_outputs_fast(
                [np.zeros((1, 1, 1, 1), dtype=np.float32)] * 5,
                conf_thr=0.3,
                iou_thr=0.4,
                class_names=["object"],
                letterbox_meta=LetterboxMeta(1, 0, 0, 1, 1, 1, 1),
                input_hw=(1, 1),
            )


if __name__ == "__main__":
    unittest.main()
