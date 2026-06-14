from research_pipeline.serve.live_backend import camera_status, health, methods


def test_live_backend_exposes_primary_and_ablation_methods() -> None:
    payload = health()
    method_payload = methods()

    assert "c1t_tcn" in payload["methods"]
    assert "c6_ensemble" in payload["methods"]
    assert any(item["id"] == "c6_ensemble" for item in method_payload["methods"])
    assert "c3" in payload["ablation_methods"]
    assert any(item["id"] == "c3" for item in method_payload["ablations"])


def test_live_backend_exposes_task_aware_interaction_mode() -> None:
    assert "c4_task_aware" in health()["interaction_modes"]


def test_camera_status_contract_is_available_without_active_camera() -> None:
    status = camera_status()
    assert {
        "running",
        "error",
        "camera_index",
        "requested_width",
        "requested_height",
        "target_fps",
        "width",
        "height",
        "capture_fps",
        "frame_age_ms",
        "backend",
    }.issubset(status)
