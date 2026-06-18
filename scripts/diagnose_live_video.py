"""Offline diagnostic: run the live pipeline on a recorded webcam video and log
the raw geometry signals the live controller uses (which the session JSONL omits).
"""

from __future__ import annotations

import sys
from collections import deque

import numpy as np

from research_pipeline.features.preprocessing import palm_scale
from research_pipeline.serve.live_backend import FrameLandmarker, LiveLandmarkGestureController, tensor_from_window

VIDEO = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Maksim Iuzhakov\Desktop\Another_one_bite\20260618_15_59_03_714.mp4"


def main() -> None:
    import cv2

    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {VIDEO}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    landmarker = FrameLandmarker()
    controller = LiveLandmarkGestureController()
    window: deque = deque(maxlen=32)

    rows = []
    FIRED: list = []
    frame_idx = 0
    valid_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        landmarks, valid, conf = landmarker.detect(frame, timestamp_ms=int(frame_idx * 1000.0 / fps))
        window.append((landmarks, valid, conf))
        valid_count += int(valid)
        tensor = tensor_from_window(window)
        stats = controller._stats(tensor)
        # run the full controller to capture fired discrete events
        from research_pipeline.models.common import prediction_from_scores
        prev_mode = controller.last_context.get("mode")
        prev_cand = controller.last_context.get("candidate_label")
        controller.update(prediction_from_scores({"no_gesture": 1.0}), tensor)
        ctx = controller.last_context
        if ctx.get("mode") == "locked" and not (prev_mode == "locked" and prev_cand == ctx.get("candidate_label")):
            FIRED.append((round(frame_idx / fps, 1), ctx.get("candidate_label")))
        # absolute gaps (normalized by palm scale) for the most recent valid frame
        ti_abs = im_abs = float("nan")
        if valid:
            scale = max(float(palm_scale(landmarks[None, :, :])[0]), 1e-6)
            ti_abs = float(np.linalg.norm(landmarks[4, :2] - landmarks[8, :2]) / scale)   # thumb-index gap
            im_abs = float(np.linalg.norm(landmarks[8, :2] - landmarks[12, :2]) / scale)  # index-middle gap
        rows.append(
            {
                "f": frame_idx,
                "t": frame_idx / fps,
                "valid": valid,
                "ti_abs": ti_abs,        # thumb-index absolute gap (pinch state)
                "im_abs": im_abs,        # index-middle absolute gap (click state)
                "click_min": stats.get("click_close_ratio_min", 1.0),
                "ti_gap_stat": stats.get("thumb_index_gap", 1.0),
                "motion": stats.get("motion", 0.0),
                "dx": stats.get("dx", 0.0),
                "dy": stats.get("dy", 0.0),
            }
        )
        frame_idx += 1
    cap.release()

    n = len(rows)
    print(f"video={VIDEO}")
    print(f"frames={n} fps={fps:.1f} valid_rate={valid_count/max(1,n):.3f}")

    def pcts(key, cond=lambda r: r["valid"]):
        vals = [r[key] for r in rows if cond(r) and not np.isnan(r[key])]
        if not vals:
            return "no data"
        a = np.array(vals)
        return f"min={a.min():.3f} p10={np.percentile(a,10):.3f} p50={np.percentile(a,50):.3f} p90={np.percentile(a,90):.3f} max={a.max():.3f}"

    print("index-middle gap (click; lower=fingers together):", pcts("im_abs"))
    print("thumb-index gap  (pinch; lower=pinched):         ", pcts("ti_abs"))
    # swipe signal: horizontal index-tip displacement over the window (palm-normalized)
    absdx = np.array([abs(r["dx"]) for r in rows if r["valid"]])
    if absdx.size:
        print(f"|dx| (swipe; current threshold max(0.16, jitter*3.8)): p50={np.percentile(absdx,50):.3f} p90={np.percentile(absdx,90):.3f} p99={np.percentile(absdx,99):.3f} max={absdx.max():.3f}")
    print()
    print("FIRED EVENTS (controller-locked) [t(s), gesture]:")
    from collections import Counter
    for t, g in FIRED:
        print(f"   {t:5.1f}s  {g}")
    print("   counts:", dict(Counter(g for _, g in FIRED)))
    print()
    # downsampled timeline so the gesture segments are visible
    print("t(s)  valid  ti_gap  im_gap     dx    motion")
    step = max(1, n // 60)
    for r in rows[::step]:
        ti = f"{r['ti_abs']:.2f}" if not np.isnan(r["ti_abs"]) else "  - "
        im = f"{r['im_abs']:.2f}" if not np.isnan(r["im_abs"]) else "  - "
        print(f"{r['t']:5.1f}  {int(r['valid'])}    {ti:>5}  {im:>5}   {r['dx']:+.3f}   {r['motion']:.3f}")


if __name__ == "__main__":
    main()
