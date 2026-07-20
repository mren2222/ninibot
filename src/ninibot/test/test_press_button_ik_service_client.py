import math
import time
from types import SimpleNamespace

import numpy as np

from ninibot.skills.press_button.ik_service_client import FineAlignIkServiceClient


class FakeFuture:
    def __init__(self, response=None, done=True, exc=None):
        self.response = response
        self.done_value = done
        self.exc = exc

    def done(self):
        return self.done_value

    def result(self):
        if self.exc is not None:
            raise self.exc
        return self.response


class FakeClient:
    def __init__(self, future, ready=True):
        self.future = future
        self.ready = ready
        self.requests = []

    def service_is_ready(self):
        return self.ready

    def call_async(self, request):
        self.requests.append(request)
        return self.future


def make_service(client, *, fallback_enabled=True, timeout_sec=0.5):
    return FineAlignIkServiceClient(
        client,
        enabled=True,
        service_name="/solve_grippertip_ik",
        timeout_sec=timeout_sec,
        fallback_enabled=fallback_enabled,
        joint6_rad=0.5,
        log_progress=lambda *args, **kwargs: None,
    )


def start_request(service):
    return service.start_request(
        {"ik_step_target": np.asarray([0.1, 0.2, 0.3])},
        target_xyz=[0.1, 0.2, 0.3],
        seed_q1234=[0.0, 0.1, 0.2, 0.3],
        lock_joint4=True,
        max_joint_step_rad=math.radians(8.0),
        ik_position_tolerance=0.01,
        joint4_min_rad=math.radians(30.0),
        joint4_max_rad=math.radians(60.0),
        fixed_joint5_rad=-math.pi / 2.0,
    )


def make_response(success=True, error=0.0, message=""):
    return SimpleNamespace(
        success=success,
        error=error,
        q1234=[1.0, 2.0, 3.0, 4.0],
        residual_xyz=[0.001, 0.002, 0.003],
        message=message,
    )


def test_start_request_builds_grippertip_ik_request():
    client = FakeClient(FakeFuture(make_response()))
    service = make_service(client)

    assert start_request(service)

    request = client.requests[0]
    assert request.target_x == 0.1
    assert request.target_y == 0.2
    assert request.target_z == 0.3
    assert list(request.seed_q1234) == [0.0, 0.1, 0.2, 0.3]
    assert request.lock_joint4 is True
    assert math.isclose(request.max_joint_step_deg, 8.0)
    assert math.isclose(request.joint4_min_deg, 30.0)
    assert math.isclose(request.joint4_max_deg, 60.0)
    assert math.isclose(request.fixed_joint5_rad, -math.pi / 2.0)
    assert request.joint6_rad == 0.5


def test_poll_accepts_near_miss_service_result():
    client = FakeClient(FakeFuture(make_response(success=False, error=0.004, message="close")))
    service = make_service(client)
    start_request(service)

    result = service.poll(near_miss_tolerance=0.005)

    assert result.kind == FineAlignIkServiceClient.MOVE
    assert result.source == "service_near_miss"
    assert result.pose == [1.0, 2.0, 3.0, 4.0, 0.5]
    assert result.error == 0.004
    assert result.q1234 == [1.0, 2.0, 3.0, 4.0]
    assert "near-miss" in result.warning


def test_poll_returns_fallback_on_timeout_when_enabled():
    client = FakeClient(FakeFuture(done=False))
    service = make_service(client, fallback_enabled=True, timeout_sec=0.1)
    start_request(service)
    service.pending_start_time = time.monotonic() - 1.0

    result = service.poll(near_miss_tolerance=0.005)

    assert result.kind == FineAlignIkServiceClient.FALLBACK
    assert result.reason == "service_timeout"
    assert service.pending_future is None


def test_poll_fails_service_error_when_fallback_disabled():
    client = FakeClient(FakeFuture(make_response(success=False, error=0.2, message="too far")))
    service = make_service(client, fallback_enabled=False)
    start_request(service)

    result = service.poll(near_miss_tolerance=0.005)

    assert result.kind == FineAlignIkServiceClient.FAILED
    assert result.reason == "IK service: too far"


def test_start_request_reports_unavailable_service():
    logs = []
    client = FakeClient(FakeFuture(make_response()), ready=False)
    service = FineAlignIkServiceClient(
        client,
        enabled=True,
        service_name="/solve_grippertip_ik",
        timeout_sec=0.5,
        fallback_enabled=True,
        joint6_rad=0.5,
        log_progress=lambda text, **kwargs: logs.append((text, kwargs)),
    )

    assert not start_request(service)
    assert not client.requests
    assert logs[0][0] == "press button: IK service not ready: /solve_grippertip_ik"
    assert logs[0][1]["period_sec"] == 0.5
