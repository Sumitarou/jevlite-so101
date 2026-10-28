"""M4a video overlay: a band at the bottom of each frame with the gate state, the judge and the
truth at that moment (from the episode's ticks). Used through eval_policy.bag_to_mp4(overlay=).
Text is ASCII (cv2 cannot draw Japanese)."""

from __future__ import annotations

import bisect

import numpy as np

# BGR versions of scripts/sim_gate_demo.STATE_COLORS (CONTINUE green, SLOW amber, STOP red)
STATE_BGR = {"CONTINUE": (92, 157, 42), "SLOW": (59, 162, 233), "STOP": (91, 73, 209)}
BAND_H = 58
RESUME_FLASH_S = 0.6


def make_overlay(ticks: list[dict], loop_t0: float, scenario: str):
    """overlay(img_bgr, stamp_s): draws the state of the last tick at or before the frame."""
    import cv2

    t_abs = [loop_t0 + k["t"] for k in ticks]
    resumes = [loop_t0 + k["t"] for k in ticks if k["gate"]["resumed"]]

    def overlay(img, stamp: float) -> None:
        k = bisect.bisect_right(t_abs, stamp + 1e-6) - 1
        h, w = img.shape[:2]
        if k < 0:
            return
        g = ticks[k]["gate"]
        color = STATE_BGR[g["state"]]
        band = img[h - BAND_H :, :]
        band[:] = (0.35 * band + 0.65 * np.array(color)).astype(band.dtype)
        d = g["decision"]
        judge = f"judge {d['choice']} p={d['conf']:.2f} age {d['age_s'] * 1e3:.0f}ms" if d \
            else "judge -"  # fmt: skip
        reason = "" if g["reason"] == "ok" else f" ({g['reason']})"
        lines = [f"GATE {g['state']}{reason}" + ("  rate-limited" if g["rate_limited"] else ""),
                 f"{judge}   truth {g['truth'] or '-'}   [{scenario}]"]  # fmt: skip
        for n, text in enumerate(lines):
            cv2.putText(img, text, (10, h - BAND_H + 22 + 24 * n), cv2.FONT_HERSHEY_SIMPLEX,
                        0.58 if n == 0 else 0.5, (255, 255, 255), 2 if n == 0 else 1,
                        cv2.LINE_AA)  # fmt: skip
        j = bisect.bisect_right(resumes, stamp + 1e-6) - 1
        if j >= 0 and stamp - resumes[j] < RESUME_FLASH_S:
            cv2.putText(img, "RESUME (policy reset)", (w - 250, h - BAND_H + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                        cv2.LINE_AA)  # fmt: skip

    return overlay
