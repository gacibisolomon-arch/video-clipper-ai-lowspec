"""
pipeline.py - Low-resource AI video clipping pipeline (CPU-only, ~8 GB RAM).

Target machine: Intel Core i5-6200U (2 physical cores / 4 threads, Skylake, AVX2),
8 GB RAM, Intel HD 520 iGPU (unused), small SSD.

Stages
  1. Fetch    : yt-dlp (capped resolution) OR a local file path (no copy).
  2. Audio    : FFmpeg -> mono 16 kHz PCM WAV (deleted right after use).
  3. ASR      : faster-whisper (CTranslate2) tiny/base, INT8, CPU, 2 threads.
  4. Score    : rule-based window scoring (questions, emotion words, loudness...).
  5. Render   : ONE FFmpeg pass per clip: cut + 9:16 reframe + burned subtitles.

Design rules
  * Nothing big ever lives in RAM: video is only touched by FFmpeg subprocesses,
    audio is read 1 second at a time, the transcript is a few KB of text.
  * Every temp file lives in one work directory that is removed in `finally`.
  * Every long operation checks a CancelToken so the UI "Cancel" button works.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# 0. THREAD LIMITS - must run BEFORE numpy / cv2 / ctranslate2 get imported.
#    i5-6200U = 2 physical cores (4 logical via Hyper-Threading). Using the
#    logical threads for heavy math just causes cache thrashing, so we pin to 2.
# ---------------------------------------------------------------------------
import os

PHYS_CORES = max(1, int(os.environ.get("CLIPPER_THREADS", "2")))
for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_var] = str(PHYS_CORES)
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")  # don't busy-spin idle threads
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import gc
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

ProgressFn = Callable[[float, str], None]


# ---------------------------------------------------------------------------
# 1. CONFIG / TYPES / CANCELLATION
# ---------------------------------------------------------------------------
@dataclass
class Config:
    # ASR
    model_size: str = "base"            # "tiny" (fastest) or "base" (more accurate)
    language: Optional[str] = None      # None = auto-detect; "en", "sw", ... = skip detection
    threads: int = PHYS_CORES
    # Download
    max_source_height: int = 720        # 480 = faster + smaller; 720 = sharper
    # Clip selection
    min_clip_s: float = 30.0
    max_clip_s: float = 60.0
    num_clips: int = 5
    # Render
    reframe: str = "crop"               # "crop" | "blur" | "pad"
    face_detect: bool = True            # Haar cascade; only used when reframe == "crop"
    burn_subs: bool = True
    out_w: int = 720
    out_h: int = 1280
    crf: int = 26                       # higher = smaller/faster, lower quality
    preset: str = "veryfast"            # ultrafast/superfast/veryfast (slower = better compression)
    # Safety
    min_free_gb: float = 3.0
    work_root: Optional[str] = None     # where temp files go (None = system temp)
    output_root: str = "outputs"


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Clip:
    start: float
    end: float
    score: float
    reasons: List[str] = field(default_factory=list)
    text: str = ""
    path: str = ""


class Cancelled(Exception):
    """Raised when the user presses Cancel."""


class CancelToken:
    def __init__(self) -> None:
        self._ev = threading.Event()

    def cancel(self) -> None:
        self._ev.set()

    @property
    def cancelled(self) -> bool:
        return self._ev.is_set()

    def check(self) -> None:
        if self._ev.is_set():
            raise Cancelled()


def _noop_progress(frac: float, msg: str) -> None:  # default progress sink
    pass


def _sub(progress: ProgressFn, lo: float, hi: float) -> ProgressFn:
    """Map a stage-local 0..1 progress onto the global [lo, hi] range."""
    return lambda f, m: progress(lo + (hi - lo) * max(0.0, min(1.0, f)), m)


def limit_torch_threads(n: int) -> None:
    """
    faster-whisper runs on CTranslate2, NOT PyTorch, so torch is normally not
    installed at all (saves ~200 MB disk and RAM). If you DO add a torch-based
    component later, this caps it - but only if torch is already loaded, so we
    never import it just to configure it.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already initialised


# ---------------------------------------------------------------------------
# 2. SAFE SUBPROCESS + PREFLIGHT
# ---------------------------------------------------------------------------
def run_cmd(cmd: List[str], cancel: CancelToken, cwd: Optional[Path] = None) -> None:
    """
    Run a command at LOW priority (keeps the laptop responsive), poll for
    cancellation, and send stderr to a temp file (a full pipe can deadlock).
    """
    if os.name == "nt":
        kwargs = {
            "creationflags": subprocess.BELOW_NORMAL_PRIORITY_CLASS
            | subprocess.CREATE_NO_WINDOW
        }
    else:
        kwargs = {}
        if shutil.which("nice"):
            cmd = ["nice", "-n", "10"] + cmd

    with tempfile.TemporaryFile() as errf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=errf,
            **kwargs,
        )
        while True:
            try:
                proc.wait(timeout=0.3)
                break
            except subprocess.TimeoutExpired:
                if cancel.cancelled:
                    proc.kill()
                    proc.wait()
                    raise Cancelled()
        if proc.returncode != 0:
            errf.seek(0)
            tail = errf.read().decode("utf-8", "replace")[-800:]
            raise RuntimeError(f"{cmd[0]} failed (code {proc.returncode}):\n{tail}")


def preflight(cfg: Config, work_dir: Path, log: ProgressFn) -> None:
    for exe in ("ffmpeg", "ffprobe"):
        if not shutil.which(exe):
            raise RuntimeError(
                f"'{exe}' not found on PATH. Install FFmpeg first (see README)."
            )
    if cfg.burn_subs:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True
        ).stdout
        if not re.search(r"\bsubtitles\b", out):
            log(0.0, "FFmpeg build has no libass/'subtitles' filter - subtitles disabled.")
            cfg.burn_subs = False
    free_gb = shutil.disk_usage(work_dir).free / 1e9
    if free_gb < cfg.min_free_gb:
        raise RuntimeError(
            f"Only {free_gb:.1f} GB free on the work drive (need {cfg.min_free_gb} GB). "
            "Free some space or set a different work_root."
        )


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    ).stdout.strip()
    try:
        return float(out)
    except ValueError:
        raise RuntimeError(f"Could not read duration of {path}")


# ---------------------------------------------------------------------------
# 3. STEP 1 - FETCH SOURCE
# ---------------------------------------------------------------------------
def fetch_source(
    source: str, work_dir: Path, cfg: Config, cancel: CancelToken, progress: ProgressFn
) -> Tuple[Path, str]:
    """Return (video_path, title). Local paths are used in place - never copied."""
    local = Path(source.strip().strip('"').strip("'"))
    if local.exists():
        return local, local.stem

    import yt_dlp  # lazy import: keeps startup light

    def hook(d: dict) -> None:
        cancel.check()  # raising here aborts the download
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            if total:
                progress(d.get("downloaded_bytes", 0) / total, "Downloading video...")

    h = cfg.max_source_height
    opts = {
        "format": f"bv*[height<={h}]+ba/b[height<={h}]/worst",
        "outtmpl": str(work_dir / "source.%(ext)s"),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 3,
        "concurrent_fragment_downloads": 1,   # 1 connection: gentle on CPU/RAM
        "progress_hooks": [hook],
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(source, download=True)
    except Exception as exc:
        if cancel.cancelled:
            raise Cancelled() from exc
        raise RuntimeError(f"Download failed: {exc}\nTip: pip install -U yt-dlp") from exc

    files = [p for p in work_dir.glob("source.*") if p.suffix not in (".part", ".ytdl")]
    if not files:
        raise RuntimeError("Download finished but no video file was found.")
    files.sort(key=lambda p: (p.suffix != ".mp4", -p.stat().st_size))
    return files[0], (info or {}).get("title") or "video"


# ---------------------------------------------------------------------------
# 4. STEP 1b - AUDIO STRIP (mono / 16 kHz / PCM16)  ~115 MB per hour
# ---------------------------------------------------------------------------
def extract_audio(video: Path, wav: Path, cancel: CancelToken) -> None:
    run_cmd(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", "-threads", "1", str(wav)],
        cancel,
    )


def loudness_per_second(wav: Path, cancel: CancelToken) -> List[float]:
    """RMS per second, streamed 1 s at a time (RAM use ~ 64 KB regardless of length)."""
    out: List[float] = []
    with wave.open(str(wav), "rb") as w:
        sr = w.getframerate()
        while True:
            raw = w.readframes(sr)
            if not raw:
                break
            a = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            out.append(float(np.sqrt(np.mean(a * a))) if a.size else 0.0)
            if len(out) % 120 == 0:
                cancel.check()
    return out


# ---------------------------------------------------------------------------
# 5. STEP 2 - TRANSCRIPTION (faster-whisper, INT8, CPU)
# ---------------------------------------------------------------------------
def transcribe(
    wav: Path, duration: float, cfg: Config, cancel: CancelToken, progress: ProgressFn
) -> List[Segment]:
    from faster_whisper import WhisperModel  # lazy import

    progress(0.0, f"Loading Whisper '{cfg.model_size}' (INT8)...")
    model = WhisperModel(
        cfg.model_size,
        device="cpu",
        compute_type="int8",       # INT8 weights: ~4x smaller RAM, fast on AVX2
        cpu_threads=cfg.threads,   # = physical cores
        num_workers=1,             # one decode stream; more would just fight for cores
    )
    try:
        seg_iter, info = model.transcribe(
            str(wav),
            language=cfg.language,
            beam_size=1,                         # greedy decoding: much faster
            best_of=1,
            temperature=0.0,                     # no costly fallback re-decodes
            condition_on_previous_text=False,    # fewer hallucination loops, less compute
            vad_filter=True,                     # skip silence (bundled Silero VAD, ONNX)
            vad_parameters={"min_silence_duration_ms": 500},
            word_timestamps=False,
        )
        out: List[Segment] = []
        for s in seg_iter:                       # generator => streamed, not batched
            cancel.check()
            text = s.text.strip()
            if text:
                out.append(Segment(float(s.start), float(s.end), text))
            progress(s.end / max(duration, 1.0), f"Transcribing... {int(s.end)}s / {int(duration)}s")
        return out
    finally:
        del model
        gc.collect()  # hand the ~150-300 MB back before rendering starts


# ---------------------------------------------------------------------------
# 6. STEP 3 - RULE-BASED CLIP SCORING (no LLM)
# ---------------------------------------------------------------------------
# Edit freely. English + a few Swahili/Sheng words; add your streamers' catchphrases.
EMOTION_WORDS = {
    "crazy", "insane", "unbelievable", "shocking", "worst", "secret", "truth",
    "lie", "lied", "liar", "scam", "money", "million", "billion", "hate", "angry",
    "scared", "afraid", "cry", "cried", "wow", "omg", "fight", "fired", "cheat",
    "cheated", "betray", "betrayed", "expose", "exposed", "dead", "kill", "broke",
    "seriously", "literally", "never", "always", "nobody", "everyone",
    # Swahili / Sheng
    "wueh", "kumbe", "kweli", "pesa", "mbaya", "hatari", "uongo", "siri",
    "jamani", "kichaa", "ujinga",
}
_WORD_RE = re.compile(r"[a-zA-Z']+")
_LAUGH_RE = re.compile(r"\b(?:ha){2,}h?\b|\blol\b|\blmao\b", re.I)
_NUM_RE = re.compile(r"\d+")
_TERMINAL = (".", "?", "!")


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def score_window(
    segs: List[Segment], start: float, end: float, loud: List[float], loud_p90: float
) -> Tuple[float, List[str]]:
    text = " ".join(s.text for s in segs)
    words = _WORD_RE.findall(text.lower())
    dur = max(end - start, 1.0)
    reasons: List[str] = []
    score = 0.0

    q = text.count("?")
    if q:
        score += min(q, 4) * 1.5
        reasons.append(f"{q} question(s)")
    ex = text.count("!")
    if ex:
        score += min(ex, 4) * 1.0
        reasons.append(f"{ex} exclamation(s)")
    kw = sum(1 for w in words if w in EMOTION_WORDS)
    if kw:
        score += min(kw, 6) * 1.2
        reasons.append(f"{kw} emotional keyword(s)")
    laugh = len(_LAUGH_RE.findall(text))
    if laugh:
        score += min(laugh, 3) * 1.0
        reasons.append("laughter")
    nums = len(_NUM_RE.findall(text))
    if nums:
        score += min(nums, 3) * 0.5
        reasons.append("numbers/stats")

    density = len(words) / dur                      # words per second
    score += _clamp((density - 1.5) / 2.0, 0, 1) * 2.0
    if density > 2.8:
        reasons.append("fast-paced speech")

    if loud and loud_p90 > 0:                       # loudness peaks
        a, b = int(start), min(int(end) + 1, len(loud))
        chunk = loud[a:b]
        if chunk:
            ratio = (sum(chunk) / len(chunk)) / loud_p90
            score += _clamp(ratio, 0, 1.5) * 2.0
            if ratio > 1.0:
                reasons.append("loud/energetic")

    # Hook: the first ~4 seconds decide whether people keep watching.
    hook_text = " ".join(s.text for s in segs if s.start < start + 4.0)
    hook_words = _WORD_RE.findall(hook_text.lower())
    if "?" in hook_text or any(w in EMOTION_WORDS for w in hook_words):
        score += 2.0
        reasons.append("strong hook")

    # Clean ending beats a mid-sentence cut.
    if segs and segs[-1].text.rstrip().endswith(_TERMINAL):
        score += 1.0
    else:
        score -= 1.0
    return score, reasons


def select_clips(
    segs: List[Segment], loud: List[float], cfg: Config, cancel: CancelToken
) -> List[Clip]:
    """Slide a window over segment boundaries, score it, keep top non-overlapping clips."""
    if not segs:
        return []
    loud_p90 = float(np.percentile(loud, 90)) if loud else 0.0
    candidates: List[Clip] = []

    for i, first in enumerate(segs):
        if i % 50 == 0:
            cancel.check()
        start = max(0.0, first.start - 0.3)
        chosen: Optional[int] = None
        for j in range(i, len(segs)):
            dur = segs[j].end - start
            if dur > cfg.max_clip_s:
                break
            if dur >= cfg.min_clip_s:
                if chosen is None:
                    chosen = j                       # first window long enough
                if segs[j].text.rstrip().endswith(_TERMINAL):
                    chosen = j                       # prefer ending on a full sentence
                    break
        if chosen is None:
            continue
        window = segs[i : chosen + 1]
        end = window[-1].end + 0.2
        sc, why = score_window(window, start, end, loud, loud_p90)
        candidates.append(
            Clip(start=start, end=end, score=round(sc, 2), reasons=why,
                 text=" ".join(s.text for s in window)[:300])
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    picked: List[Clip] = []
    for c in candidates:
        if all(c.end + 5 < p.start or c.start > p.end + 5 for p in picked):
            picked.append(c)
        if len(picked) >= cfg.num_clips:
            break
    return picked


# ---------------------------------------------------------------------------
# 7. STEP 4 - SUBTITLES + FACE FRAMING + FFMPEG FILTER GRAPH
# ---------------------------------------------------------------------------
def _srt_ts(t: float) -> str:
    t = max(t, 0.0)
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt(segs: List[Segment], c_start: float, c_end: float, path: Path) -> int:
    """
    SRT with times relative to the clip start. Long segments are split into
    5-word cues (timing spread by character count) so captions stay short and
    readable on a phone, without needing word-level timestamps (which cost CPU).
    """
    cues: List[Tuple[float, float, str]] = []
    for s in segs:
        if s.end <= c_start or s.start >= c_end:
            continue
        words = s.text.split()
        if not words:
            continue
        chunks = [" ".join(words[i : i + 5]) for i in range(0, len(words), 5)]
        total = sum(len(c) for c in chunks) or 1
        span = s.end - s.start
        t = s.start
        for c in chunks:
            t0, t1 = t, t + span * len(c) / total
            t = t1
            a, b = max(t0, c_start) - c_start, min(t1, c_end) - c_start
            if b - a > 0.15:
                cues.append((a, b, c))
    with open(path, "w", encoding="utf-8") as f:
        for n, (a, b, txt) in enumerate(cues, 1):
            f.write(f"{n}\n{_srt_ts(a)} --> {_srt_ts(b)}\n{txt}\n\n")
    return len(cues)


def detect_face_x(
    video: Path, c_start: float, c_end: float, cancel: CancelToken
) -> Tuple[int, int, Optional[float]]:
    """
    Cheap framing: sample 7 frames across the clip, run an OpenCV Haar cascade on
    a 480px-wide grayscale copy, return the MEDIAN horizontal face position
    (0..1). One static crop per clip - no per-frame tracking, no ML weights.
    Returns (src_w, src_h, face_x_fraction_or_None).
    """
    import cv2

    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(video))
    try:
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        xs: List[float] = []
        n = 7
        for k in range(n):
            cancel.check()
            cap.set(cv2.CAP_PROP_POS_MSEC, (c_start + (c_end - c_start) * (k + 0.5) / n) * 1000)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            scale = min(1.0, 480.0 / frame.shape[1])
            small = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))
            if len(faces):
                x, _, w, _ = max(faces, key=lambda f: f[2] * f[3])  # biggest face
                xs.append((x + w / 2) / small.shape[1])
        face = float(np.median(xs)) if len(xs) >= 2 else None
        return W, H, face
    finally:
        cap.release()


def build_vf(
    cfg: Config, src_w: int, src_h: int, face_x: Optional[float], srt_name: Optional[str]
) -> str:
    """Build the single -vf string: reframe to 9:16, then burn subtitles."""
    ow, oh = cfg.out_w, cfg.out_h
    mode = cfg.reframe
    portrait_src = src_w and src_h and src_w <= src_h * 9 / 16
    if mode == "crop" and portrait_src:
        mode = "pad"  # source already vertical - just fit it

    if mode == "crop":
        if face_x is not None and src_w and src_h:
            cw = int(src_h * 9 / 16) // 2 * 2
            x = int(_clamp(face_x * src_w - cw / 2, 0, src_w - cw))
            chain = f"crop={cw}:{src_h}:{x}:0,scale={ow}:{oh}:flags=bilinear"
        else:  # centered crop via FFmpeg expressions (works for any source size)
            chain = f"crop=w='trunc(ih*9/16/2)*2':h=ih,scale={ow}:{oh}:flags=bilinear"
    elif mode == "blur":
        # Cheap blurred background: shrink to 144x256, blur, scale back up.
        chain = (
            "split[a][b];"
            f"[a]scale=144:256:force_original_aspect_ratio=increase,crop=144:256,"
            f"boxblur=4:1,scale={ow}:{oh}:flags=bilinear[bg];"
            f"[b]scale={ow}:-2:flags=bilinear[fg];"
            "[bg][fg]overlay=(W-w)/2:(H-h)/2"
        )
    else:  # pad
        chain = f"scale={ow}:-2:flags=bilinear,pad={ow}:{oh}:(ow-iw)/2:(oh-ih)/2:black"

    if srt_name:
        style = (
            "FontName=Arial,FontSize=34,Bold=1,PrimaryColour=&H00FFFFFF,"
            "OutlineColour=&H00000000,BorderStyle=1,Outline=3,Shadow=0,"
            "Alignment=2,MarginV=140"  # bottom-centre, above Reels/Shorts UI
        )
        chain += f",subtitles={srt_name}:original_size={ow}x{oh}:force_style='{style}'"
    return chain + ",format=yuv420p"


def render_clip(
    src: Path, clip: Clip, segs: List[Segment], rank: int,
    out_dir: Path, work_dir: Path, cfg: Config, cancel: CancelToken,
) -> Path:
    clip_dir = work_dir / f"clip{rank}"
    clip_dir.mkdir(parents=True, exist_ok=True)
    try:
        srt_name = None
        if cfg.burn_subs:
            # Relative filename + cwd=clip_dir sidesteps Windows "C:\..." escaping in filtergraphs.
            if build_srt(segs, clip.start, clip.end, clip_dir / "subs.srt"):
                srt_name = "subs.srt"
                shutil.copy(clip_dir / "subs.srt", out_dir / f"clip_{rank:02d}.srt")

        W = H = 0
        face = None
        if cfg.reframe == "crop" and cfg.face_detect:
            W, H, face = detect_face_x(src, clip.start, clip.end, cancel)

        vf = build_vf(cfg, W, H, face, srt_name)
        name = f"clip_{rank:02d}_{int(clip.start // 60):02d}m{int(clip.start % 60):02d}s.mp4"
        out_path = (out_dir / name).resolve()

        run_cmd(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-ss", f"{clip.start:.2f}", "-i", str(src.resolve()),
             "-t", f"{clip.end - clip.start:.2f}",
             "-vf", vf,
             "-c:v", "libx264", "-preset", cfg.preset, "-crf", str(cfg.crf),
             "-threads", str(cfg.threads),
             "-c:a", "aac", "-b:a", "128k", "-ac", "2",
             "-movflags", "+faststart", str(out_path)],
            cancel, cwd=clip_dir,
        )
        return out_path
    finally:
        shutil.rmtree(clip_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 8. ORCHESTRATOR
# ---------------------------------------------------------------------------
def run_pipeline(
    source: str,
    cfg: Optional[Config] = None,
    cancel: Optional[CancelToken] = None,
    progress: ProgressFn = _noop_progress,
) -> dict:
    cfg = cfg or Config()
    cancel = cancel or CancelToken()
    limit_torch_threads(cfg.threads)

    work_root = Path(cfg.work_root) if cfg.work_root else Path(tempfile.gettempdir())
    work_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="clipper_", dir=work_root))
    try:
        preflight(cfg, work_dir, progress)

        # 1) fetch -------------------------------------------------------
        video, title = fetch_source(source, work_dir, cfg, cancel, _sub(progress, 0.00, 0.15))
        duration = probe_duration(video)
        cancel.check()

        # 2) audio -> ASR -> loudness -> delete WAV immediately ----------
        progress(0.15, "Extracting 16 kHz mono audio...")
        wav = work_dir / "audio.wav"
        try:
            extract_audio(video, wav, cancel)
            segs = transcribe(wav, duration, cfg, cancel, _sub(progress, 0.18, 0.62))
            progress(0.62, "Measuring loudness...")
            loud = loudness_per_second(wav, cancel)
        finally:
            wav.unlink(missing_ok=True)   # free ~115 MB/hour right now

        if not segs:
            raise RuntimeError("No speech detected - nothing to clip.")

        # 3) score -------------------------------------------------------
        progress(0.64, "Scoring transcript for viral moments...")
        clips = select_clips(segs, loud, cfg, cancel)
        if not clips:
            raise RuntimeError(
                "No windows matched the length limits. Try a wider min/max clip length."
            )

        # 4) render (sequential, one FFmpeg at a time) -------------------
        slug = re.sub(r"[^\w\-]+", "_", title).strip("_")[:50] or "video"
        out_dir = Path(cfg.output_root) / f"{slug}_{time.strftime('%Y%m%d_%H%M%S')}"
        out_dir.mkdir(parents=True, exist_ok=True)
        render_progress = _sub(progress, 0.66, 1.0)
        for rank, clip in enumerate(clips, 1):
            cancel.check()
            render_progress((rank - 1) / len(clips), f"Rendering clip {rank}/{len(clips)}...")
            clip.path = str(render_clip(video, clip, segs, rank, out_dir, work_dir, cfg, cancel))
        render_progress(1.0, "Done.")

        result = {"title": title, "output_dir": str(out_dir), "clips": [asdict(c) for c in clips]}
        (out_dir / "clips.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), "utf-8")
        return result
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)  # downloaded source, WAV, SRTs - all gone
