"""Low-resource AI video clipping pipeline for i5-6200U / 8GB RAM laptops."""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OMP_DYNAMIC", "FALSE")

import re, gc, json, time, subprocess, contextlib
from dataclasses import dataclass, field
from pathlib import Path

import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

import cv2

_CASCADE = None
def _get_cascade():
    global _CASCADE
    if _CASCADE is None:
        _CASCADE = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _CASCADE


@dataclass
class Clip:
    start: float; end: float; score: float; reason: str; text: str = ""

@dataclass
class JobConfig:
    url: str
    out_dir: Path = Path("clips")
    model_size: str = "base"
    clip_min: int = 30; clip_max: int = 60; max_clips: int = 5
    vertical: bool = True
    burn_subtitles: bool = True
    face_track: bool = False
    keep_temp: bool = False

@dataclass
class JobState:
    progress: float = 0.0; stage: str = "idle"; msg: str = ""
    clips: list = field(default_factory=list)
    cancelled: bool = False; error: str = ""
    def check_cancel(self):
        if self.cancelled: raise JobCancelled()

class JobCancelled(Exception): pass


def _run(cmd: list[str]) -> None:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True,
                            encoding="utf-8", errors="replace")
    _, err = proc.communicate(timeout=10800)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {err[-800:]}")

def download_lowres(url: str, dest: Path, state: JobState) -> Path:
    state.stage, state.msg = "download", "Fetching video (<=720p)"
    _run(["yt-dlp",
          "-f", "bv*[height<=720]+ba/b[height<=720]/b",
          "--merge-output-format", "mp4", "--no-playlist",
          "--restrict-filenames",
          "-o", str(dest.with_suffix(".%(ext)s")), url])
    state.check_cancel()
    for p in dest.parent.glob(dest.stem + ".*"):
        if p.suffix not in (".wav", ".srt", ".ass", ".json"):
            return p
    raise FileNotFoundError("yt-dlp produced no video file")

def strip_audio(video: Path, wav: Path, state: JobState) -> Path:
    state.stage, state.msg = "audio", "Extracting mono 16kHz audio"
    _run(["ffmpeg", "-y", "-i", str(video), "-vn",
          "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(wav)])
    state.check_cancel()
    return wav


def transcribe(wav: Path, cfg: JobConfig, state: JobState):
    from faster_whisper import WhisperModel
    state.stage, state.msg = "transcribe", f"Loading {cfg.model_size} (INT8/CPU)"
    state.progress = 0.15; state.check_cancel()

    model = WhisperModel(cfg.model_size, device="cpu",
                         compute_type="int8",
                         cpu_threads=2, num_workers=1)
    total = _wav_duration(wav)
    segments, info = model.transcribe(
        str(wav), language=None, vad_filter=True,
        beam_size=1, word_timestamps=False,
        condition_on_previous_text=False)

    segs, done = [], 0.0
    for seg in segments:
        state.check_cancel()
        segs.append(seg); done = seg.end
        state.progress = 0.15 + 0.30 * min(done / max(total, 1.0), 1.0)
    del model; gc.collect()
    return segs, info

def _wav_duration(wav: Path) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(wav)],
            capture_output=True, text=True, timeout=30)
        return float(out.stdout.strip())
    except Exception:
        return 0.0


EMOTION_KEYWORDS = {
    "secret", "truth", "never", "always", "shocking", "mistake", "wrong",
    "crazy", "insane", "amazing", "incredible", "learned", "changed",
    "nobody", "everyone", "important", "stop", "fail", "win", "money",
    "free", "hack", "best", "worst", "first", "last"}
QUESTION_BONUS, KEYWORD_BONUS, DENSITY_BONUS = 3.0, 2.0, 1.5

def score_clips(segments, cfg: JobConfig, state: JobState) -> list[Clip]:
    state.stage, state.msg = "score", "Scoring highlights (rule-based)"
    state.progress = 0.50; state.check_cancel()

    words = [(s.start, w.lower()) for s in segments for w in s.text.split()]
    if not words: return []

    t0, t_end = words[0][0], words[-1][0] + 1
    cands, t = [], t0
    while t + cfg.clip_min <= t_end:
        state.check_cancel()
        chunk = [(ts, w) for ts, w in words if t <= ts < t + cfg.clip_max]
        if len(chunk) >= 8:
            text = " ".join(w for _, w in chunk)
            score, reasons = 0.0, []
            n_q = text.count("?")
            if n_q:
                score += QUESTION_BONUS * min(n_q, 3); reasons.append(f"{n_q}x question")
            n_k = sum(1 for k in EMOTION_KEYWORDS if k in text)
            if n_k:
                score += KEYWORD_BONUS * min(n_k, 4); reasons.append(f"{n_k} keywords")
            dur = chunk[-1][0] - chunk[0][0] + 1e-6
            if len(chunk) / dur > 2.6:
                score += DENSITY_BONUS; reasons.append("high energy")
            gaps = _max_gap([ts for ts, _ in chunk])
            if gaps > 2.5:
                score -= 2.0; reasons.append("dead air")
            cands.append(Clip(t, t + min(cfg.clip_max, dur + 1), score,
                              ", ".join(reasons) or "context", text[:300]))
        t += 5.0

    cands.sort(key=lambda c: -c.score)
    chosen = []
    for c in cands:
        if len(chosen) >= cfg.max_clips: break
        if all(abs(c.start - o.start) > cfg.clip_min for o in chosen):
            chosen.append(c)
    chosen.sort(key=lambda c: c.start)
    state.progress = 0.60
    return chosen

def _max_gap(times) -> float:
    return max((b - a for a, b in zip(times, times[1:])), default=0.0)


def _ts(s: float) -> str:
    h, r = divmod(int(s), 3600); m, sec = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{sec:02d},000"

def write_srt(clips: list[Clip], out: Path) -> Path:
    with open(out, "w", encoding="utf-8") as f:
        for i, c in enumerate(clips, 1):
            f.write(f"{i}\n{_ts(c.start)} --> {_ts(c.end)}\n{c.reason}\n\n")
    return out

def _detect_face_x(video: Path, t: float) -> float | None:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read(); cap.release()
    if not ok: return None
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (320, int(320 * h / w)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    faces = _get_cascade().detectMultiScale(gray, 1.2, 4, minSize=(40, 40))
    if len(faces) == 0: return None
    x, _y, fw, _fh = max(faces, key=lambda f: f[2] * f[3])
    return (x + fw / 2) / small.shape[1]

def render_clip(video, clip, idx, srt, cfg, state) -> Path:
    state.stage, state.msg = "render", f"Rendering clip {idx}"
    state.check_cancel()
    out = cfg.out_dir / f"clip_{idx:02d}_{int(clip.start)}s.mp4"
    vf = []
    if cfg.vertical:
        cx = 0.5
        if cfg.face_track:
            fx = _detect_face_x(video, clip.start + 1.0)
            if fx is not None:
                cx = min(max(fx, 0.28125), 0.71875)
        x_expr = f"(iw-{1080/1920:.4f}*ih)*{cx:.3f}"
        vf.append(f"scale=-2:1920,crop=1080:1920:x='{x_expr}':y=0")
    else:
        vf.append("scale=1280:-2")
    if cfg.burn_subtitles:
        srt_esc = str(srt).replace("\\", "/").replace(":", "\\:")
        vf.append(f"subtitles='{srt_esc}'")
    vf.append("format=yuv420p")
    _run(["ffmpeg", "-y", "-ss", f"{clip.start:.2f}", "-to", f"{clip.end:.2f}",
          "-i", str(video), "-vf", ",".join(vf),
          "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
          "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(out)])
    return out


def run_job(cfg: JobConfig, state: JobState) -> list[Path]:
    workdir = Path("temp"); workdir.mkdir(exist_ok=True)
    stem = f"job_{int(time.time())}"
    video, wav, srt = (workdir / f"{stem}_src.mp4",
                       workdir / f"{stem}.wav", workdir / f"{stem}.srt")
    outputs = []
    try:
        state.progress = 0.02
        src = download_lowres(cfg.url, video, state); state.progress = 0.10
        strip_audio(src, wav, state);                 state.progress = 0.15
        segments, _ = transcribe(wav, cfg, state)
        clips = score_clips(segments, cfg, state)
        if not clips:
            state.msg = "No highlights found"; return []

        write_srt(clips, srt)
        manifest = []
        for i, c in enumerate(clips, 1):
            state.check_cancel()
            p = render_clip(src, c, i, srt, cfg, state)
            outputs.append(p)
            manifest.append({"file": p.name, "start": c.start, "end": c.end,
                             "score": c.score, "reason": c.reason, "text": c.text})
            state.progress = 0.60 + 0.38 * i / len(clips)

        with open(cfg.out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        state.progress = 1.0
        state.stage, state.msg = "done", f"{len(outputs)} clips ready"
        return outputs
    finally:
        if not cfg.keep_temp:
            for f in (video, wav, srt):
                with contextlib.suppress(OSError):
                    f.unlink(missing_ok=True)
        gc.collect()


class JobRunner:
    def __init__(self):
        self.state = JobState(); self._thread = None
    def start(self, cfg: JobConfig) -> None:
        if self._thread and self._thread.is_alive():
            raise RuntimeError("A job is already running")
        self.state = JobState(stage="queued", msg="Starting…")
        import threading
        self._thread = threading.Thread(target=self._guard, args=(cfg,),
                                        daemon=True)
        self._thread.start()
    def _guard(self, cfg):
        try:
            run_job(cfg, self.state)
        except JobCancelled:
            self.state.stage, self.state.msg = "cancelled", "Cancelled by user"
        except Exception as e:
            self.state.stage, self.state.error = "error", str(e)
            self.state.msg = "Failed"
    def cancel(self):
        self.state.cancelled = True
