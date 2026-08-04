# Copyright (c) 2026 清影/123twtd
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

from .ipm_M import FastDetectionIPM


Direction = Literal["L", "R", "none"]
HumanDirection = Literal["left", "right", "still"]
ObstacleLabel = Literal["human", "car"]
PlannerMode = Literal[
    "normal",
    "avoiding",
    "confirming",
    "waiting",
    "recentering",
    "collecting_gold",
]

_CANDIDATE_OFFSET_RATIOS = (-1.0, -0.8, -0.4, 0.0, 0.4, 0.8, 1.0)
_HUMAN_PREDICT_GAIN = 2.0
_HUMAN_MOTION_THRESHOLD_CM = 1.0
_LATERAL_FORWARD_RATIO = 2.0
_SAFE_DISTANCE_EPS_CM = 1.0


def transform_dets_by_ipm(dets, ipm, frame_shape=None):
    """Project human/car/gold detections to the IPM pixel coordinate system.

    ``ipm`` may be either ``FastDetectionIPM`` or the image ``FastIPM`` from
    ``ipm_utils``. In the latter case the calibrated matrix is reused through
    ``FastDetectionIPM.from_ipm``.
    """

    if isinstance(ipm, FastDetectionIPM):
        projector = ipm
    else:
        projector = getattr(ipm, "_detection_ipm_projector", None)
        if not isinstance(projector, FastDetectionIPM):
            projector = FastDetectionIPM.from_ipm(ipm)
            try:
                ipm._detection_ipm_projector = projector
            except (AttributeError, TypeError):
                pass
    return projector.process(dets, frame_shape=frame_shape)


def ipm_detections_to_ar_objects(
    ipm_dets,
    *,
    ipm_w: int,
    ipm_h: int,
    track_width_cm: float,
    track_depth_cm: float,
) -> list["ARObject"]:
    """Convert ``ipm_M.IPMDetection`` objects from pixels to centimeters."""

    width_denominator = max(int(ipm_w) - 1, 1)
    height_denominator = max(int(ipm_h) - 1, 1)
    cm_per_px_x = float(track_width_cm) / width_denominator
    cm_per_px_y = float(track_depth_cm) / height_denominator
    center_x_px = (int(ipm_w) - 1) * 0.5
    objects: list[ARObject] = []

    for det in ipm_dets or ():
        try:
            px = float(det.center_x_px)
            py = float(det.center_y_px)
            label = str(det.class_name)
            score = float(det.score)
        except (AttributeError, TypeError, ValueError):
            continue

        x_cm = (px - center_x_px) * cm_per_px_x
        y_cm = max(0.0, (int(ipm_h) - 1 - py) * cm_per_px_y)
        objects.append(
            ARObject(
                label=label,
                x_cm=x_cm,
                y_cm=y_cm,
                score=score,
                in_lane=abs(x_cm) <= float(track_width_cm) * 0.5,
            )
        )
    return objects


def detections_to_ar_objects(
    dets,
    ipm,
    *,
    frame_shape=None,
    track_width_cm: float = 60.0,
    track_depth_cm: float = 90.0,
) -> tuple[list["ARObject"], bool]:
    """Project raw detections and convert them to planner objects in one call."""

    ipm_dets, success = transform_dets_by_ipm(
        dets,
        ipm,
        frame_shape=frame_shape,
    )
    if not success:
        return [], False
    out_w, out_h = ipm.out_size
    return (
        ipm_detections_to_ar_objects(
            ipm_dets,
            ipm_w=out_w,
            ipm_h=out_h,
            track_width_cm=track_width_cm,
            track_depth_cm=track_depth_cm,
        ),
        True,
    )


@dataclass(slots=True)
class Occupancy:
    """Lateral space occupied by a human or car in vehicle coordinates."""

    label: ObstacleLabel
    left_cm: float
    right_cm: float
    y_cm: float
    direction: HumanDirection = "still"


@dataclass(slots=True)
class ARObject:
    """Detected object in centimeters; x is left-negative, y is forward."""

    label: str
    x_cm: float
    y_cm: float
    score: float = 0.5
    in_lane: bool = True


@dataclass(slots=True)
class ARDecision:
    """Planner output consumed by the line-following control layer."""

    offset_cm: float = 0.0
    speed_scale: float = 1.0
    speed_limit_cm_s: int | None = None
    stop: bool = False
    fork_prefer: Direction = "none"
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ARMapConfig:
    """Human, car and gold planner parameters in centimeters."""

    track_width_cm: float = 60.0
    robot_width_cm: float = 22.0
    human_width_cm: float = 8.0
    car_width_cm: float = 30.0   # 障碍车宽估大，逼 planner 多避
    safety_margin_cm: float = 3.0  # 不动，加大反而缩小可用偏移范围
    max_offset_cm: float = 16.0

    human_predict_y_cm: float = 70.0
    car_avoid_y_cm: float = 85  # 70->85，提前避障，不超过视野深度90
    confirm_frames: int = 2
    obstacle_clear_hold_frames: int = 2
    recenter_frames: int = 3
    human_offset_step_cm: float = 2.0
    car_offset_step_cm: float = 5.0  # 步长加大，近处车也能快速到满偏

    human_confirm_speed_scale: float = 0.30
    human_avoid_speed_scale: float = 0.45
    car_avoid_speed_scale: float = 0.60
    obstacle_clear_speed_scale: float = 0.65
    recenter_speed_scale: float = 0.75

    gold_pick_y_cm: float = 80.0
    gold_offset_step_cm: float = 2.0


class QuadraticObstacleStrategy:
    """Choose a collision-free lateral offset using a quadratic cost."""

    def __init__(
        self,
        *,
        lookahead_cm: float,
        lateral_forward_ratio: float = _LATERAL_FORWARD_RATIO,
        soft_clearance_cm: float = 5.0,
        continuity_weight: float = 1.0,
        center_weight: float = 0.15,
        clearance_weight: float = 4.0,
        human_direction_penalty: float = 100.0,
    ) -> None:
        self.lookahead_cm = max(float(lookahead_cm), 1e-6)
        self.lateral_forward_ratio = max(float(lateral_forward_ratio), 0.0)
        self.soft_clearance_cm = max(float(soft_clearance_cm), 0.0)
        self.continuity_weight = max(float(continuity_weight), 0.0)
        self.center_weight = max(float(center_weight), 0.0)
        self.clearance_weight = max(float(clearance_weight), 0.0)
        self.human_direction_penalty = max(float(human_direction_penalty), 0.0)

    def choose_offset(
        self,
        occupancies: Iterable[Occupancy],
        candidates: Iterable[float],
        current_offset: float,
        vehicle_half_width: float,
    ) -> float | None:
        bands = tuple(occupancies)
        scored: list[tuple[float, float, float, float]] = []
        for raw_offset in candidates:
            offset = float(raw_offset)
            required_forward = abs(offset - current_offset) * self.lateral_forward_ratio
            vehicle_left = offset - vehicle_half_width
            vehicle_right = offset + vehicle_half_width
            score = (
                self.continuity_weight * (offset - current_offset) ** 2
                + self.center_weight * offset**2
            )
            rejected = False

            for occupancy in bands:
                if required_forward > occupancy.y_cm:
                    rejected = True
                    break
                if self._overlaps(vehicle_left, vehicle_right, occupancy):
                    rejected = True
                    break

                gap = self._clearance(vehicle_left, vehicle_right, occupancy)
                proximity = self._proximity_weight(occupancy.y_cm)
                clearance_error = max(0.0, self.soft_clearance_cm - gap)
                score += self.clearance_weight * proximity * clearance_error**2
                score += self._human_direction_cost(offset, occupancy, proximity)

            if not rejected:
                scored.append((score, abs(offset - current_offset), abs(offset), offset))

        return None if not scored else min(scored)[-1]

    def _proximity_weight(self, y_cm: float) -> float:
        normalized = max(0.0, 1.0 - y_cm / self.lookahead_cm)
        return normalized**2

    def _human_direction_cost(
        self,
        offset: float,
        occupancy: Occupancy,
        proximity: float,
    ) -> float:
        wrong_side = (
            occupancy.label == "human"
            and (
                (occupancy.direction == "right" and offset > 0.0)
                or (occupancy.direction == "left" and offset < 0.0)
            )
        )
        if not wrong_side:
            return 0.0
        return self.human_direction_penalty * (0.5 + proximity)

    @staticmethod
    def _overlaps(
        vehicle_left: float,
        vehicle_right: float,
        occupancy: Occupancy,
    ) -> bool:
        return vehicle_right >= occupancy.left_cm and vehicle_left <= occupancy.right_cm

    @staticmethod
    def _clearance(
        vehicle_left: float,
        vehicle_right: float,
        occupancy: Occupancy,
    ) -> float:
        if vehicle_right < occupancy.left_cm:
            return occupancy.left_cm - vehicle_right
        if vehicle_left > occupancy.right_cm:
            return vehicle_left - occupancy.right_cm
        return 0.0


class SimpleARMapPlanner:
    """Plan offsets and speed for human/car avoidance and gold collection."""

    def __init__(
        self,
        config: ARMapConfig | None = None,
        behavior_modules=None,
    ) -> None:
        self.config = config or ARMapConfig()
        self._validate_config()
        self.behavior_modules = list(behavior_modules or ())
        self.mode: PlannerMode = "normal"
        self.commanded_offset_cm = 0.0
        self.candidate_offset_cm: float | None = None
        self.safe_frames = 0
        self.clear_frames = 0
        self.active_obstacle_kind = "obstacle"
        self.prev_human_x_cm: float | None = None
        self.prev_human_y_cm: float | None = None
        self.obstacle_strategy = QuadraticObstacleStrategy(
            lookahead_cm=max(
                self.config.human_predict_y_cm,
                self.config.car_avoid_y_cm,
            )
        )

    def _validate_config(self) -> None:
        cfg = self.config
        positive = {
            "track_width_cm": cfg.track_width_cm,
            "robot_width_cm": cfg.robot_width_cm,
            "human_width_cm": cfg.human_width_cm,
            "car_width_cm": cfg.car_width_cm,
        }
        invalid = [name for name, value in positive.items() if float(value) <= 0.0]
        if invalid:
            raise ValueError(f"planner dimensions must be positive: {', '.join(invalid)}")
        for name in (
            "human_confirm_speed_scale",
            "human_avoid_speed_scale",
            "car_avoid_speed_scale",
            "obstacle_clear_speed_scale",
            "recenter_speed_scale",
        ):
            value = float(getattr(cfg, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

    def plan(self, objects: Iterable[ARObject]) -> ARDecision:
        valid_objects = tuple(
            obj
            for obj in objects
            if float(obj.score) > 0.0 and float(obj.y_cm) >= 0.0
        )
        humans = self._objects_of_label(
            valid_objects,
            "human",
            self.config.human_predict_y_cm,
        )
        cars = self._objects_of_label(
            valid_objects,
            "car",
            self.config.car_avoid_y_cm,
        )

        occupancies = self.build_obstacle_occupancies(humans, cars)
        if occupancies:
            decision = self.plan_obstacles(occupancies, bool(humans), bool(cars))
        else:
            decision = self.plan_without_obstacles(valid_objects)

        for module in self.behavior_modules:
            module.apply_decision(decision)

        nearest_human = self._nearest(valid_objects, "human")
        if nearest_human is None:
            self.prev_human_x_cm = None
            self.prev_human_y_cm = None
        else:
            self.prev_human_x_cm = nearest_human.x_cm
            self.prev_human_y_cm = nearest_human.y_cm
        return decision

    def plan_obstacles(
        self,
        occupancies: list[Occupancy],
        has_human: bool,
        has_car: bool,
    ) -> ARDecision:
        previous_kind = self.active_obstacle_kind
        obstacle_kind = (
            "mixed" if has_human and has_car else "human" if has_human else "car"
        )
        selected_offset = self.obstacle_strategy.choose_offset(
            occupancies,
            self.candidate_offsets(),
            self.commanded_offset_cm,
            self.vehicle_clearance_half_width(),
        )
        self.clear_frames = 0

        if selected_offset is None:
            self.mode = "waiting"
            self.active_obstacle_kind = obstacle_kind
            self._reset_candidate()
            return self._stop_decision(f"{obstacle_kind}_no_safe_corridor")

        already_avoiding_human = (
            self.mode == "avoiding" and previous_kind in {"human", "mixed"}
        )
        needs_confirmation = has_human and not already_avoiding_human
        needs_confirmation = needs_confirmation or self.mode in {"confirming", "waiting"}
        if needs_confirmation and not self._candidate_confirmed(selected_offset):
            self.mode = "confirming"
            self.active_obstacle_kind = obstacle_kind
            decision = ARDecision(
                offset_cm=self.commanded_offset_cm,
                speed_scale=self.config.human_confirm_speed_scale,
            )
            decision.reasons.append(
                f"{obstacle_kind}_wait_safe_{self.safe_frames}/{self.config.confirm_frames}"
            )
            decision.reasons.append(
                f"speed_scale_{self.config.human_confirm_speed_scale:.2f}"
            )
            return decision

        self.mode = "avoiding"
        self.active_obstacle_kind = obstacle_kind
        self._reset_candidate()
        max_step = (
            self.config.human_offset_step_cm
            if has_human
            else self.config.car_offset_step_cm
        )
        self.commanded_offset_cm = self.move_offset_toward(
            self.commanded_offset_cm,
            selected_offset,
            max_step,
        )
        speed_scale = (
            self.config.human_avoid_speed_scale
            if has_human
            else self.config.car_avoid_speed_scale
        )
        decision = ARDecision(
            offset_cm=self.commanded_offset_cm,
            speed_scale=speed_scale,
        )
        decision.reasons.append(f"{obstacle_kind}_quadratic_avoid")
        decision.reasons.append(f"target_offset_{selected_offset:.1f}cm")
        decision.reasons.append(f"speed_scale_{speed_scale:.2f}")
        return decision

    def plan_without_obstacles(
        self,
        objects: tuple[ARObject, ...],
    ) -> ARDecision:
        if self.mode in {"avoiding", "confirming", "waiting"}:
            hold = self._hold_after_obstacle_clear()
            if hold is not None:
                return hold
            self.mode = "recentering"

        if self.mode == "recentering":
            return self._recenter_decision("obstacle_clear")

        golds = self._objects_of_label(objects, "gold", self.config.gold_pick_y_cm)
        collectible = tuple(gold for gold in golds if self.gold_is_collectible(gold))
        gold = min(collectible, key=lambda obj: (obj.y_cm, -obj.score), default=None)

        if self.mode == "collecting_gold":
            if gold is not None:
                return self.plan_gold(gold)
            self.mode = "recentering"
            return self._recenter_decision("gold_clear")

        if gold is not None and self.commanded_offset_cm == 0.0:
            return self.plan_gold(gold)

        if golds:
            nearest_gold = min(golds, key=lambda obj: (obj.y_cm, -obj.score))
            decision = ARDecision(offset_cm=self.commanded_offset_cm)
            decision.reasons.append(self.gold_rejection_reason(nearest_gold))
            return decision

        return ARDecision(offset_cm=self.commanded_offset_cm)

    def build_obstacle_occupancies(
        self,
        humans: tuple[ARObject, ...],
        cars: tuple[ARObject, ...],
    ) -> list[Occupancy]:
        occupancies: list[Occupancy] = []
        nearest_human = humans[0] if humans else None
        for human in humans:
            occupancies.append(
                self.build_human_occupancy(
                    human,
                    predict_motion=human is nearest_human,
                )
            )
        car_half_width = self.config.car_width_cm / 2.0
        occupancies.extend(
            Occupancy(
                label="car",
                left_cm=car.x_cm - car_half_width,
                right_cm=car.x_cm + car_half_width,
                y_cm=car.y_cm,
            )
            for car in cars
        )
        return occupancies

    def build_human_occupancy(
        self,
        human: ARObject,
        *,
        predict_motion: bool = True,
    ) -> Occupancy:
        cfg = self.config
        dx = self.human_step_x(human) if predict_motion else 0.0
        direction = self.human_direction(dx)
        predicted_x = human.x_cm + dx * _HUMAN_PREDICT_GAIN
        proximity = self.clamp(
            1.0 - human.y_cm / max(cfg.human_predict_y_cm, 1e-6),
            0.0,
            1.0,
        )
        dynamic_margin = cfg.safety_margin_cm * proximity**2
        human_half_width = cfg.human_width_cm / 2.0
        return Occupancy(
            label="human",
            left_cm=min(human.x_cm, predicted_x) - human_half_width - dynamic_margin,
            right_cm=max(human.x_cm, predicted_x) + human_half_width + dynamic_margin,
            y_cm=human.y_cm,
            direction=direction,
        )

    def plan_gold(self, gold: ARObject) -> ARDecision:
        limit = self.offset_limit()
        target_offset = self.clamp(gold.x_cm, -limit, limit)
        self.commanded_offset_cm = self.move_offset_toward(
            self.commanded_offset_cm,
            target_offset,
            self.config.gold_offset_step_cm,
        )
        self.mode = "collecting_gold"
        decision = ARDecision(offset_cm=self.commanded_offset_cm)
        decision.reasons.append("gold_pick")
        decision.reasons.append(f"target_offset_{target_offset:.1f}cm")
        return decision

    def gold_is_collectible(self, gold: ARObject) -> bool:
        if not gold.in_lane:
            return False
        track_half_width = self.config.track_width_cm / 2.0
        reachable = self.offset_limit() + self.config.robot_width_cm / 2.0
        return abs(gold.x_cm) <= min(track_half_width, reachable)

    def gold_rejection_reason(self, gold: ARObject) -> str:
        if not gold.in_lane or abs(gold.x_cm) > self.config.track_width_cm / 2.0:
            return "gold_outside_track_ignore"
        return "gold_unreachable_ignore"

    def human_step_x(self, human: ARObject) -> float:
        if self.prev_human_x_cm is None or self.prev_human_y_cm is None:
            return 0.0
        if abs(human.y_cm - self.prev_human_y_cm) > self.config.human_predict_y_cm / 3.0:
            return 0.0
        dx = human.x_cm - self.prev_human_x_cm
        max_step = max(self.config.human_width_cm, self.config.safety_margin_cm)
        return self.clamp(dx, -max_step, max_step)

    @staticmethod
    def human_direction(dx_cm: float) -> HumanDirection:
        if dx_cm > _HUMAN_MOTION_THRESHOLD_CM:
            return "right"
        if dx_cm < -_HUMAN_MOTION_THRESHOLD_CM:
            return "left"
        return "still"

    def _candidate_confirmed(self, selected_offset: float) -> bool:
        frames = max(1, self.config.confirm_frames)
        if self.candidate_offset_cm is None:
            self.candidate_offset_cm = selected_offset
            self.safe_frames = 1
        elif abs(selected_offset - self.candidate_offset_cm) <= _SAFE_DISTANCE_EPS_CM:
            self.safe_frames += 1
        else:
            self.candidate_offset_cm = selected_offset
            self.safe_frames = 1
        return self.safe_frames >= frames

    def _hold_after_obstacle_clear(self) -> ARDecision | None:
        self.clear_frames += 1
        frames = max(0, self.config.obstacle_clear_hold_frames)
        if self.clear_frames > frames:
            return None
        reason = f"{self.active_obstacle_kind}_clear_hold_{self.clear_frames}/{frames}"
        if self.mode == "waiting":
            return self._stop_decision(reason)
        decision = ARDecision(
            offset_cm=self.commanded_offset_cm,
            speed_scale=self.config.obstacle_clear_speed_scale,
        )
        decision.reasons.extend(
            [reason, f"speed_scale_{self.config.obstacle_clear_speed_scale:.2f}"]
        )
        return decision

    def _recenter_decision(self, reason: str) -> ARDecision:
        self.commanded_offset_cm = self.move_offset_toward(
            self.commanded_offset_cm,
            0.0,
            self.recenter_step(),
        )
        decision = ARDecision(
            offset_cm=self.commanded_offset_cm,
            speed_scale=self.config.recenter_speed_scale,
        )
        decision.reasons.extend(
            [reason, f"speed_scale_{self.config.recenter_speed_scale:.2f}"]
        )
        if self.commanded_offset_cm == 0.0:
            self.mode = "normal"
            self.clear_frames = 0
        else:
            decision.reasons.append(f"recenter_{self.commanded_offset_cm:.1f}cm")
        return decision

    def _stop_decision(self, reason: str) -> ARDecision:
        decision = ARDecision(
            offset_cm=self.commanded_offset_cm,
            speed_scale=0.0,
            stop=True,
        )
        decision.reasons.append(reason)
        return decision

    def _reset_candidate(self) -> None:
        self.candidate_offset_cm = None
        self.safe_frames = 0

    @staticmethod
    def move_offset_toward(current: float, target: float, max_step: float) -> float:
        step = max(0.0, float(max_step))
        delta = max(-step, min(step, float(target) - float(current)))
        return float(current) + delta

    @staticmethod
    def _objects_of_label(
        objects: Iterable[ARObject],
        label: str,
        max_y_cm: float,
    ) -> tuple[ARObject, ...]:
        return tuple(
            sorted(
                (
                    obj
                    for obj in objects
                    if obj.label == label and obj.y_cm <= max_y_cm
                ),
                key=lambda obj: (obj.y_cm, -obj.score),
            )
        )

    @staticmethod
    def _nearest(objects: Iterable[ARObject], label: str) -> ARObject | None:
        return min(
            (obj for obj in objects if obj.label == label),
            key=lambda obj: (obj.y_cm, -obj.score),
            default=None,
        )

    def candidate_offsets(self) -> tuple[float, ...]:
        limit = self.offset_limit()
        if limit <= 0.0:
            return (0.0,)
        return tuple(limit * ratio for ratio in _CANDIDATE_OFFSET_RATIOS)

    def vehicle_clearance_half_width(self) -> float:
        return self.config.robot_width_cm / 2.0 + self.config.safety_margin_cm

    def offset_limit(self) -> float:
        track_limit = self.config.track_width_cm / 2.0 - self.vehicle_clearance_half_width()
        return max(0.0, min(self.config.max_offset_cm, track_limit))

    def recenter_step(self) -> float:
        frames = max(1, self.config.recenter_frames)
        return max(1.0, self.offset_limit() / frames)

    @staticmethod
    def clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))
