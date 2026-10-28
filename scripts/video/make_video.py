"""Draft of the submission video from existing bags (no new simulator runs).

    uv run --project train python scripts/video/make_video.py [--only NAME ...] [--out DIR]

Reads configs/video.yaml (sections, captions, rates), the camera images of the bags with rosbags
(train env) and the gate state of each tick (ticks.jsonl). Writes outputs/video/<stamp>/:
NN_<name>.mp4 and NN_<name>_first.png per section, draft.mp4 (concatenated), timeline.json and a
copy of the config. --only with --out re-renders some sections of an existing folder and
concatenates again. Japanese text is drawn with PIL (font from fc-match)."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "train")]

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from scripts.sim.gate_video import STATE_BGR  # noqa: E402
from scripts.video import video_core as vc  # noqa: E402

SIM_OUT = ROOT / "outputs" / "sim"
STATE_RGB = {k: tuple(reversed(v)) for k, v in STATE_BGR.items()}
SEVERITY = {"CONTINUE": 0, "SLOW": 1, "STOP": 2}
PANEL_X = 960  # clip / still: camera 960x720 on the left, the state panel on the right
PAD = 16


# ---------------------------------------------------------------- resources


class Fonts:
    def __init__(self, cfg: dict):
        self.paths = {k: self._match(v) for k, v in cfg["font"].items()}
        self.cache: dict = {}

    @staticmethod
    def _match(pattern: str) -> tuple[str, int]:
        r = subprocess.run(["fc-match", "-f", "%{file}\n%{index}", pattern], capture_output=True,
                           text=True, check=True)  # fmt: skip
        path, idx = r.stdout.split("\n")
        if "CJK" not in Path(path).name and "Noto" not in Path(path).name:
            raise RuntimeError(f"no Japanese font for {pattern!r} (fc-match gave {path})")
        return path, int(idx or 0)

    def __call__(self, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        key = (size, bold)
        if key not in self.cache:
            path, idx = self.paths["bold" if bold else "regular"]
            self.cache[key] = ImageFont.truetype(path, size, index=idx)
        return self.cache[key]


class Source:
    """Camera frames (loop time) and ticks of one episode bag."""

    def __init__(self, spec: dict, topic: str):
        from export_intruder_frames import read_bag

        run = SIM_OUT / spec["run"]
        self.bag = run / "bags" / spec["bag"]
        rows = [json.loads(x) for x in (run / "results.jsonl").read_text().splitlines()]
        row = next(r for r in rows if Path(r["bag"]).name == spec["bag"])
        self.t0 = row.get("loop_t0") or row["sim_start"]
        self.row = row
        stamps, self.images, _ = read_bag(self.bag, topic, "/intruder/pose")
        self.stamps = list(np.asarray(stamps) - self.t0)
        self.ticks = [json.loads(x) for x in (self.bag / "ticks.jsonl").read_text().splitlines()]
        self.tick_t = [k["t"] for k in self.ticks]
        self.resumes = [k["t"] for k in self.ticks if k.get("gate", {}).get("resumed")]

    def image(self, t: float) -> np.ndarray:
        return self.images[vc.frame_index(self.stamps, t)]

    def gate(self, t: float) -> dict | None:
        k = vc.last_at(self.tick_t, t)
        return self.ticks[k].get("gate") if k >= 0 else None

    def resumed_recently(self, t: float, window: float = 0.8) -> bool:
        j = vc.last_at(self.resumes, t)
        return j >= 0 and t - self.resumes[j] < window and self.resumes[j] > 0.6  # not start-up

    def tcp_speed_cm_s(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from scripts.sim.gate_loop import tcp_speed
        from scripts.sim.robustness_core import TcpFK, load_configs

        _, sim_cfg = load_configs()
        t = np.array(self.tick_t, float)
        q = np.array([k["gate"]["q_now"][:5] for k in self.ticks], float)
        state = np.array([k["gate"]["state"] for k in self.ticks])
        return t, tcp_speed(t, TcpFK(sim_cfg)(q)) * 100, state


# ---------------------------------------------------------------- drawing helpers


class Painter:
    def __init__(self, cfg: dict, fonts: Fonts):
        self.cfg, self.f = cfg, fonts
        self.W, self.H = cfg["output"]["width"], cfg["output"]["height"]
        self.c = {k: tuple(v) if isinstance(v, list) else v for k, v in cfg["colors"].items()}
        self.L = cfg["labels"]
        x0, y0, x1, y1 = cfg["camera"]["crop"]
        self.crop = (x0, y0, x1, y1)

    def canvas(self) -> Image.Image:
        return Image.new("RGB", (self.W, self.H), self.c["bg"])

    def camera(self, img: np.ndarray, size: tuple[int, int]) -> Image.Image:
        x0, y0, x1, y1 = self.crop
        return Image.fromarray(img[y0:y1, x0:x1]).resize(size, Image.LANCZOS)

    def to_view(self, xy, box) -> tuple[float, float]:
        """Source pixel -> canvas pixel for a camera view drawn in box (x, y, w, h)."""
        x0, y0, x1, y1 = self.crop
        bx, by, bw, bh = box
        return (bx + (xy[0] - x0) * bw / (x1 - x0), by + (xy[1] - y0) * bh / (y1 - y0))

    def text(self, d: ImageDraw.ImageDraw, xy, s: str, size: int, fill=None, bold=False,
             anchor="la") -> None:  # fmt: skip
        d.text(xy, s, font=self.f(size, bold), fill=fill or self.c["text"], anchor=anchor)

    def wrapped(self, d, x, y, s: str, size: int, width: int, fill=None, bold=False,
                spacing: float = 1.35) -> int:  # fmt: skip
        """Draw wrapped text from (x, y) down; returns the y after the last line. One character
        of width is kept free for punctuation that hangs at a line end."""
        font = self.f(size, bold)
        for line in vc.wrap_text(s, width - size, lambda u: d.textlength(u, font=font)):
            d.text((x, y), line, font=font, fill=fill or self.c["text"])
            y += round(size * spacing)
        return y

    def wrapped_height(self, d, s: str, size: int, width: int, spacing: float = 1.35) -> int:
        font = self.f(size)
        return len(vc.wrap_text(s, width - size, lambda u: d.textlength(u, font=font))) * round(
            size * spacing)

    def note_box(self, d, x: int, y: int, s: str, width: int, size: int = 17) -> int:
        """The note on what the judge is, outlined so it is not missed; returns the bottom y."""
        h = self.wrapped_height(d, s, size, width - 20)
        d.rounded_rectangle((x, y, x + width, y + h + 14), 8, outline=self.c["accent"], width=2)
        self.wrapped(d, x + 10, y + 7, s, size, width - 20, fill=self.c["text"])
        return y + h + 14

    def speed_badge(self, d, xy, speed: float, size: int = 30) -> None:
        """Big playback rate on the camera view whenever it is not 1 (paused is not shown)."""
        if speed == 0 or abs(speed - 1) < 1e-9:
            return
        s = vc.speed_label(speed, self.L["speed"])
        font = self.f(size, True)
        w = d.textlength(s, font=font)
        x, y = xy
        d.rounded_rectangle((x, y, x + w + 24, y + size + 18), 8, fill=(0, 0, 0, 170))
        d.text((x + 12, y + 7), s, font=font, fill=self.c["accent"])

    def caption_block(self, d, captions: list[str], x: int, bottom: int, width: int,
                      size: int = 22, top: int | None = None) -> None:  # fmt: skip
        """Captions on a slightly lighter box, stacked upward from bottom (or down from top)."""
        if not captions:
            return
        s = "\n".join(captions)
        h = self.wrapped_height(d, s, size, width - 24)
        y0 = bottom - h - 24 if top is None else top
        d.rounded_rectangle((x, y0, x + width, y0 + h + 24), 10, fill=(52, 58, 70))
        self.wrapped(d, x + 12, y0 + 12, s, size, width - 24)

    def state_lines(self, g: dict | None) -> tuple[tuple, str, list[str]]:
        """(color, state label, detail lines) of a gate tick for the band / panel."""
        L = self.L
        if g is None:
            return self.c["no_gate"], "-", []
        dec = g["decision"]
        if dec:
            judge = (f"{L['judge']}：{L['choices'][dec['choice']]}  p={dec['conf']:.2f}  "
                     f"{L['age']} {dec['age_s'] * 1e3:.0f} ms")  # fmt: skip
        else:
            judge = f"{L['judge']}：-"
        lines = [judge, f"{L['truth']}：{L['zones'].get(g['truth'] or 'none', '-')}"]
        reason = L["reasons"].get(g["reason"], g["reason"])
        if not reason and SEVERITY[g["state"]] > SEVERITY.get(g["requested"], 0):
            reason = L["hold"]  # G6: waiting for the looser judgement to last resume_hold_s
        if reason:
            lines.append(f"{L['reason']}：{reason}")
        return STATE_RGB[g["state"]], L["states"][g["state"]], lines


# ---------------------------------------------------------------- sections


def render_title(p: Painter, sec: dict, n: int):
    for _ in range(n):
        im = p.canvas()
        d = ImageDraw.Draw(im)
        cx = p.W // 2
        p.text(d, (cx, 250), sec["title"], 52, bold=True, anchor="mm")
        p.text(d, (cx, 325), sec["subtitle"], 28, fill=p.c["sub"], anchor="mm")
        for i, line in enumerate(sec.get("lines", [])):
            p.text(d, (cx, 420 + 46 * i), line, 30, fill=p.c["accent"] if i else None,
                   anchor="mm")  # fmt: skip
        if sec.get("footer"):
            p.text(d, (cx, p.H - 50), sec["footer"], 20, fill=p.c["sub"], anchor="mm")
        yield im


def render_card(p: Painter, sec: dict, n: int, fps: float):
    for i in range(n):
        t = i / fps
        im = p.canvas()
        d = ImageDraw.Draw(im)
        p.text(d, (80, 70), sec["heading"], 44, bold=True)
        y = 170
        for line in sec["lines"]:
            if t >= line.get("at", 0):
                d.ellipse((84, y + 12, 96, y + 24), fill=p.c["accent"])
                y = p.wrapped(d, 116, y, line["text"], 30, p.W - 200) + 26
        if sec.get("footer"):
            p.wrapped(d, 80, p.H - 80, sec["footer"], 20, p.W - 160, fill=p.c["sub"])
        yield im


def panel_base(p: Painter, d, heading: str) -> int:
    d.rectangle((PANEL_X, 0, p.W, p.H), fill=p.c["panel"])
    return p.wrapped(d, PANEL_X + PAD, PAD, heading, 26, p.W - PANEL_X - 2 * PAD, bold=True) + 8


def render_still(p: Painter, sec: dict, src: Source, n: int, fps: float):
    cam = p.camera(src.image(sec["t"]), (PANEL_X, p.H))
    box = (0, 0, PANEL_X, p.H)
    for i in range(n):
        t = i / fps
        im = p.canvas()
        im.paste(cam, (0, 0))
        d = ImageDraw.Draw(im, "RGBA")
        for a in sec.get("annotations", []):
            if t < a["at"]:
                continue
            tx, ty = p.to_view(a["target"], box)
            lx, ly = p.to_view(a["label"], box)
            d.line((lx, ly, tx, ty), fill=p.c["accent"], width=4)
            d.ellipse((tx - 8, ty - 8, tx + 8, ty + 8), outline=p.c["accent"], width=4)
            font = p.f(28, True)
            w = d.textlength(a["text"], font=font)
            bx, by = min(max(lx - w / 2 - 12, 8), PANEL_X - w - 32), ly - 22
            d.rounded_rectangle((bx, by, bx + w + 24, by + 44), 8, fill=(0, 0, 0, 190))
            d.text((bx + 12, by + 6), a["text"], font=font, fill=p.c["accent"])
        panel_base(p, d, sec["heading"])
        p.caption_block(d, vc.active_captions(sec.get("captions", []), t), PANEL_X + PAD,
                        p.H - PAD, p.W - PANEL_X - 2 * PAD, size=26)  # fmt: skip
        yield im


def m3e_chart(p: Painter, spec: dict, size: tuple[int, int]) -> Image.Image:
    """Pushing x vs block start x (M3e start suite): flat outside the demo range."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    path, _ = p.f.paths["regular"]
    font_manager.fontManager.addfont(path)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
    rep = SIM_OUT / spec["report"]
    rows = [r for r in csv.DictReader(open(rep / "start.csv")) if r["condition"] == "center"
            or r["condition"].startswith("x")]  # fmt: skip
    slopes = json.loads((rep / "report.json").read_text())["x_slopes"]
    x = np.array([float(r["x_cm"]) for r in rows])
    y = np.array([float(r["tcp_x_push_cm"]) for r in rows])
    ok = np.array([r["success"] == "True" for r in rows])
    dpi = 100
    fig, ax = plt.subplots(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi)
    bg = tuple(np.array(p.c["panel"]) / 255)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.axvspan(-2.5, 2.5, color="#2e86de", alpha=0.15, lw=0)
    ax.plot([-5.5, 5.5], [-5.5, 5.5], ls=":", color="#aab", lw=1)
    for key, color in (("inside", "#2e86de"), ("outside_minus", "#f39c12"),
                       ("outside_plus", "#f39c12")):  # fmt: skip
        s = slopes[key]
        xs = np.array(s["x_range_m"]) * 100
        ax.plot(xs, s["slope"] * xs + s["intercept_m"] * 100, color=color, lw=2.5)
    ax.scatter(x[ok], y[ok], s=28, color="white", zorder=3)
    ax.scatter(x[~ok], y[~ok], s=40, marker="x", color="#d1495b", zorder=3)
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-5.5, 5.5)
    ax.set_xlabel("ブロックの初期位置 x（cm）", color="white", fontsize=10)
    ax.set_title(spec["title"], color="white", fontsize=11)
    ax.text(0, 4.6, spec["inside"], color="#7fb8ff", fontsize=9.5, ha="center")
    ax.text(0, -5.1, spec["outside"], color="#f7b955", fontsize=9.5, ha="center")
    ax.tick_params(colors="white", labelsize=8)
    for s in ax.spines.values():
        s.set_color("#667")
    fig.tight_layout(pad=0.4)
    fig.canvas.draw()
    img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    return img


def render_clip(p: Painter, sec: dict, src: Source, fps: float):
    tl = vc.build_timeline(sec["segments"], fps, sec.get("pre_hold_s", 0), sec.get("hold_s", 0))
    pw = p.W - PANEL_X - 2 * PAD
    chart = m3e_chart(p, sec["chart"], (pw, 250)) if sec.get("chart") else None
    for i, (t, speed) in enumerate(tl):
        im = p.canvas()
        im.paste(p.camera(src.image(t), (PANEL_X, p.H)), (0, 0))
        d = ImageDraw.Draw(im, "RGBA")
        p.speed_badge(d, (16, 16), speed)
        x = PANEL_X + PAD
        y = panel_base(p, d, sec["heading"])
        g = src.gate(t) if sec.get("gate") else None
        if sec.get("gate"):
            color, label, lines = p.state_lines(g)
            d.rounded_rectangle((x, y, x + pw, y + 64), 10, fill=color)
            p.text(d, (x + pw / 2, y + 32), label, 30, bold=True, anchor="mm")
            y += 76
            for line in lines:
                y = p.wrapped(d, x, y, line, 18, pw)
            if src.resumed_recently(t):
                p.text(d, (x, y + 4), p.L["resume"], 20, fill=p.c["accent"], bold=True)
            y += 32
        else:
            d.rounded_rectangle((x, y, x + pw, y + 50), 10, fill=p.c["no_gate"])
            p.text(d, (x + pw / 2, y + 25), p.L["no_gate"], 20, bold=True, anchor="mm")
            y += 62
        p.text(d, (x, y), f"{p.L['sim_time']} {max(t, 0):5.2f} s", 20)
        y += 30
        odd = speed not in (0, 1)
        p.text(d, (x, y), f"{p.L['playback']} {vc.speed_label(speed, p.L['speed'])}", 20,
               fill=p.c["accent"] if odd else None, bold=odd)  # fmt: skip
        y += 36
        if sec.get("note"):
            y = p.note_box(d, x, y, sec["note"], pw) + 10
        if chart is not None:
            im.paste(chart, (x, y))
            y += chart.height
        p.caption_block(d, vc.active_captions(sec.get("captions", []), i / fps), x, p.H - PAD, pw)
        yield im


def speed_graph(p: Painter, sec: dict, series: list, size: tuple[int, int]):
    """Static graph image and a function t -> x pixel (for the cursor)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    path, _ = p.f.paths["regular"]
    font_manager.fontManager.addfont(path)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
    gcfg = sec["graph"]
    dpi = 100
    fig, ax = plt.subplots(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi)
    bg = tuple(np.array(p.c["panel"]) / 255)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    t, _, state = series[1]
    slow = state == "SLOW"
    edges = np.flatnonzero(np.diff(np.r_[0, slow.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2], strict=True):
        ax.axvspan(t[a], t[b - 1], color=np.array(STATE_RGB["SLOW"]) / 255, alpha=0.25, lw=0,
                   label=gcfg["slow_label"] if a == edges[0] else None)  # fmt: skip
    (tb, vb, _), (ta, va, _) = series
    ax.plot(tb, vb, color=p.c["before"], ls="--", lw=1.6, label=gcfg["legend"][0])
    ax.plot(ta, va, color=p.c["after"], lw=2.0, label=gcfg["legend"][1])
    ax.set_xlim(*gcfg["t_range"])
    ax.set_ylim(0, gcfg["y_max"])
    ax.set_title(gcfg["title"], color="white", fontsize=11, loc="left")
    ax.set_xlabel(p.L["sim_time"] + "（s）", color="white", fontsize=9, labelpad=1)
    ax.tick_params(colors="white", labelsize=8)
    for s in ax.spines.values():
        s.set_color("#667")
    leg = ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=9, facecolor=bg,
                    edgecolor="#667", ncol=3, borderaxespad=0.2)  # fmt: skip
    for txt in leg.get_texts():
        txt.set_color("white")
    fig.tight_layout(pad=0.3)
    fig.canvas.draw()
    img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    (x0, y0), (x1, y1) = ax.transData.transform([(gcfg["t_range"][0], 0),
                                                 (gcfg["t_range"][1], gcfg["y_max"])])  # fmt: skip
    plt.close(fig)
    h = img.height

    def tx(tt: float) -> float:
        a, b = gcfg["t_range"]
        return x0 + (min(max(tt, a), b) - a) / (b - a) * (x1 - x0)

    return img, tx, (h - y1, h - y0)


def render_compare(p: Painter, sec: dict, srcs: list[Source], fps: float):
    tl = vc.build_timeline(sec["segments"], fps, sec.get("pre_hold_s", 0), sec.get("hold_s", 0))
    cw, ch, top = 600, 450, 52
    xs = [20, p.W - 20 - cw]
    band_y, band_h = top + ch, 44
    gy = band_y + band_h + 8
    gw = 780
    graph, tx, (gy0, gy1) = speed_graph(p, sec, [s.tcp_speed_cm_s() for s in srcs],
                                        (gw, p.H - gy - 8))  # fmt: skip
    for i, (t, speed) in enumerate(tl):
        im = p.canvas()
        d = ImageDraw.Draw(im, "RGBA")
        p.text(d, (20, 12), sec["heading"], 26, bold=True)
        s_lab = f"{p.L['sim_time']} {max(t, 0):5.2f} s   {p.L['playback']} " \
            f"{vc.speed_label(speed, p.L['speed'])}"  # fmt: skip
        p.text(d, (p.W - 20, 14), s_lab, 20, fill=p.c["accent"] if speed not in (0, 1) else None,
               anchor="ra", bold=speed not in (0, 1))  # fmt: skip
        for x, src, title in zip(xs, srcs, sec["titles"], strict=True):
            im.paste(p.camera(src.image(t), (cw, ch)), (x, top))
            p.speed_badge(d, (x + cw - 190, top + 10), speed, size=24)
            d.rounded_rectangle((x + 10, top + 10, x + 30 + d.textlength(title, font=p.f(24, True)),
                                 top + 50), 8, fill=(0, 0, 0, 190))  # fmt: skip
            p.text(d, (x + 20, top + 15), title, 24, bold=True, fill=p.c["accent"])
            color, label, lines = p.state_lines(src.gate(t))
            d.rectangle((x, band_y, x + cw, band_y + band_h), fill=color)
            p.text(d, (x + 12, band_y + band_h / 2), label, 22, bold=True, anchor="lm")
            p.text(d, (x + 190, band_y + band_h / 2), lines[0] if lines else "", 16, anchor="lm")
            if src.resumed_recently(t):
                p.text(d, (x + cw - 10, band_y + band_h / 2), p.L["resume"], 16, anchor="rm")
        im.paste(graph, (20, gy))
        cx = 20 + tx(t)
        d.line((cx, gy + gy0, cx, gy + gy1), fill=p.c["accent"], width=2)
        bx = 20 + gw + 14
        bw = p.W - bx - 20
        cap_top = p.note_box(d, bx, gy, sec["note"], bw) + 6 if sec.get("note") else gy
        p.caption_block(d, vc.active_captions(sec.get("captions", []), i / fps), bx, p.H, bw,
                        size=19, top=cap_top)  # fmt: skip
        yield im


# ---------------------------------------------------------------- encoding


def encode(frames, path: Path, cfg: dict) -> tuple[int, Image.Image | None]:
    o = cfg["output"]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{o['width']}x{o['height']}", "-r", str(o["fps"]), "-i", "-",
           "-c:v", "libx264", "-preset", o["preset"], "-crf", str(o["crf"]),
           "-pix_fmt", "yuv420p", "-r", str(o["fps"]), "-movflags", "+faststart",
           str(path)]  # fmt: skip
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n, first = 0, None
    try:
        for im in frames:
            if first is None:
                first = im.copy()
            proc.stdin.write(im.convert("RGB").tobytes())
            n += 1
    finally:
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed for {path}")
    return n, first


def concat(parts: list[Path], out: Path) -> None:
    lst = out.with_suffix(".txt")
    lst.write_text("".join(f"file '{q.name}'\n" for q in parts))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", str(lst), "-c", "copy", "-movflags", "+faststart", str(out)],
                   check=True)  # fmt: skip
    lst.unlink()


def probe_s(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                        "csv=p=0", str(path)],
                       capture_output=True, text=True, check=True)  # fmt: skip
    return float(r.stdout.strip())


def git_commit() -> dict:
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="render the submission video draft")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "video.yaml")
    ap.add_argument("--only", nargs="+", help="section names to (re)render")
    ap.add_argument("--out", type=Path, help="existing output folder (with --only)")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    vc.validate(cfg)
    fps = cfg["output"]["fps"]
    names = [s["name"] for s in cfg["sections"]]
    if args.only and not set(args.only) <= set(names):
        ap.error(f"unknown section in --only (known: {names})")
    out = args.out or vc.new_stamp_dir(ROOT / cfg["output"]["base"])
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out / "config_copy.yaml")
    p = Painter(cfg, Fonts(cfg))
    cache: dict[str, Source] = {}

    def source(key: str) -> Source:
        if key not in cache:
            cache[key] = Source(cfg["sources"][key], cfg["camera"]["topic"])
        return cache[key]

    parts, info = [], []
    for idx, sec in enumerate(cfg["sections"], 1):
        path = out / f"{idx:02d}_{sec['name']}.mp4"
        parts.append(path)
        n_exp = vc.section_frames(sec, fps)
        if args.only and sec["name"] not in args.only:
            if not path.exists():
                raise SystemExit(f"{path} is missing; render it too")
            continue
        kind = sec["kind"]
        if kind == "title":
            frames = render_title(p, sec, n_exp)
        elif kind == "card":
            frames = render_card(p, sec, n_exp, fps)
        elif kind == "still":
            frames = render_still(p, sec, source(sec["source"]), n_exp, fps)
        elif kind == "clip":
            frames = render_clip(p, sec, source(sec["source"]), fps)
        else:
            frames = render_compare(p, sec, [source(k) for k in sec["sources"]], fps)
        n, first = encode(frames, path, cfg)
        if n != n_exp:
            raise RuntimeError(f"{sec['name']}: {n} frames, expected {n_exp}")
        first.save(out / f"{idx:02d}_{sec['name']}_first.png")
        print(f"{path.name}: {n} frames ({n / fps:.2f} s)", flush=True)
        keys = [sec["source"]] if "source" in sec else sec.get("sources", [])
        info.append({"section": sec["name"], "frames": n, "seconds": n / fps,
                     "sources": [cfg["sources"][k] for k in keys]})  # fmt: skip
    draft = out / "draft.mp4"
    concat(parts, draft)
    durations = {q.name: probe_s(q) for q in parts}
    total = probe_s(draft)
    tl_path = out / "timeline.json"
    old = json.loads(tl_path.read_text()) if tl_path.exists() else {}
    rendered = {x["section"]: x for x in old.get("rendered", [])} | {x["section"]: x for x in info}
    tl_path.write_text(json.dumps({
        "config": str(args.config.relative_to(ROOT)) if args.config.is_relative_to(ROOT)
        else args.config.name,
        "git": git_commit(), "draft_s": total, "expected_s": vc.total_duration_s(cfg),
        "parts_s": durations, "rendered": list(rendered.values()),
    }, ensure_ascii=False, indent=1))  # fmt: skip
    print(f"draft: {draft} ({total:.2f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
