import math
import time
from dataclasses import dataclass

from ninibot_interfaces.srv import SolveGrippertipIk


@dataclass(frozen=True)
class FineAlignIkServiceResult:
    kind: str
    context: object = None
    pose: list | None = None
    source: str = ""
    reason: str = ""
    residual_xyz: object = None
    error: float | None = None
    q1234: object = None
    warning: str = ""


class FineAlignIkServiceClient:
    WAITING = "waiting"
    MOVE = "move"
    FALLBACK = "fallback"
    FAILED = "failed"

    @staticmethod
    def create_ros_client(node, service_name):
        return node.create_client(SolveGrippertipIk, service_name)

    def __init__(
        self,
        client,
        *,
        enabled,
        service_name,
        timeout_sec,
        fallback_enabled,
        joint6_rad,
        log_progress,
    ):
        self.client = client
        self.enabled = bool(enabled)
        self.service_name = str(service_name)
        self.timeout_sec = max(0.0, float(timeout_sec))
        self.fallback_enabled = bool(fallback_enabled)
        self.joint6_rad = float(joint6_rad)
        self.log_progress = log_progress
        self.reset()

    def reset(self):
        self.pending_future = None
        self.pending_start_time = None
        self.pending_context = None

    def start_request(
        self,
        context,
        *,
        target_xyz,
        seed_q1234,
        lock_joint4,
        max_joint_step_rad,
        ik_position_tolerance,
        joint4_min_rad,
        joint4_max_rad,
        fixed_joint5_rad,
    ):
        if not self.enabled or self.client is None:
            return False
        if not self.client.service_is_ready():
            self.log_progress(
                f"press button: IK service not ready: {self.service_name}",
                period_sec=0.5,
            )
            return False

        request = SolveGrippertipIk.Request()
        request.target_x = float(target_xyz[0])
        request.target_y = float(target_xyz[1])
        request.target_z = float(target_xyz[2])
        request.seed_q1234 = [float(value) for value in seed_q1234]
        request.lock_joint4 = bool(lock_joint4)
        request.max_joint_step_deg = math.degrees(float(max_joint_step_rad))
        request.ik_position_tolerance = float(ik_position_tolerance)
        request.joint4_min_deg = math.degrees(float(joint4_min_rad))
        request.joint4_max_deg = math.degrees(float(joint4_max_rad))
        request.fixed_joint5_rad = float(fixed_joint5_rad)
        request.joint6_rad = self.joint6_rad

        self.pending_future = self.client.call_async(request)
        self.pending_start_time = time.monotonic()
        self.pending_context = context
        return True

    def poll(self, near_miss_tolerance):
        future = self.pending_future
        context = self.pending_context
        if future is None or context is None:
            return FineAlignIkServiceResult(
                self.FAILED,
                reason="missing pending IK service request",
            )

        now = time.monotonic()
        elapsed = now - (self.pending_start_time or now)
        if not future.done():
            if elapsed <= self.timeout_sec:
                self.log_progress(
                    "press button: FINE_ALIGN_IK_WAIT "
                    f"{elapsed:.2f}s/{self.timeout_sec:.2f}s"
                )
                return FineAlignIkServiceResult(self.WAITING, context=context)
            self.reset()
            if self.fallback_enabled:
                return FineAlignIkServiceResult(
                    self.FALLBACK,
                    context=context,
                    reason="service_timeout",
                )
            return FineAlignIkServiceResult(
                self.FAILED,
                context=context,
                reason="IK service timeout",
            )

        self.reset()
        try:
            response = future.result()
        except Exception as exc:
            if self.fallback_enabled:
                return FineAlignIkServiceResult(
                    self.FALLBACK,
                    context=context,
                    reason=f"service_exception:{exc}",
                )
            return FineAlignIkServiceResult(
                self.FAILED,
                context=context,
                reason=f"IK service exception: {exc}",
            )

        response_error = float(response.error)
        if len(response.q1234) >= 4 and (
            response.success
            or (
                math.isfinite(response_error)
                and response_error <= near_miss_tolerance
            )
        ):
            source = "service"
            warning = ""
            if not response.success:
                source = "service_near_miss"
                warning = (
                    "press button: accepting near-miss IK service result "
                    f"error={response_error * 100.0:.2f}cm "
                    f"near_miss_tol={near_miss_tolerance * 100.0:.2f}cm "
                    f"message={response.message}"
                )
            pose = [float(value) for value in response.q1234[:4]] + [self.joint6_rad]
            return FineAlignIkServiceResult(
                self.MOVE,
                context=context,
                pose=pose,
                source=source,
                residual_xyz=response.residual_xyz,
                error=response_error,
                q1234=response.q1234[:4],
                warning=warning,
            )

        reason = response.message or "service_failed"
        if self.fallback_enabled:
            return FineAlignIkServiceResult(
                self.FALLBACK,
                context=context,
                reason=reason,
            )
        return FineAlignIkServiceResult(
            self.FAILED,
            context=context,
            reason=f"IK service: {reason}",
        )
