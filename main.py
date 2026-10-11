"""
main.py - Minimal UI (Gradio) + optional CLI for the low-spec clipper.

    python main.py                      -> web UI at http://127.0.0.1:7860
    python main.py "<url or file>"      -> CLI, no UI (lighter still)

IMPORTANT: `import pipeline` must stay FIRST - it pins OpenMP/MKL thread counts
before numpy / gradio / ctranslate2 are loaded.
"""
import pipeline  # noqa: F401  (sets thread limits on import)
from pipeline import Cancelled, CancelToken, Config, run_pipeline

import argparse
import queue
import threading
from pathlib import Path

# One job at a time: two jobs on a dual-core laptop would just slow each other down.
_JOB_LOCK = threading.Lock()
_CURRENT = {"token": None}


def _cfg_from_ui(model, lang, min_len, max_len, n, reframe, face, subs, height) -> Config:
    return Config(
        model_size=model,
        language=None if lang == "auto" else lang,
        min_clip_s=float(min_len),
        max_clip_s=float(max(max_len, min_len + 5)),
        num_clips=int(n),
        reframe=reframe,
        face_detect=bool(face),
        burn_subs=bool(subs),
        max_source_height=int(height),
    )


# ---------------------------------------------------------------------------
# Web UI
# ---------------------------------------------------------------------------
def build_ui():
    import gradio as gr

    def run_job(source, model, lang, min_len, max_len, n, reframe, face, subs, height,
                progress=gr.Progress()):
        source = (source or "").strip()
        if not source:
            raise gr.Error("Paste a video URL or a local file path.")
        if not _JOB_LOCK.acquire(blocking=False):
            raise gr.Error("A job is already running - cancel it or wait.")

        token = CancelToken()
        _CURRENT["token"] = token
        events: "queue.Queue" = queue.Queue()
        box: dict = {}
        cfg = _cfg_from_ui(model, lang, min_len, max_len, n, reframe, face, subs, height)

        def worker():
            try:
                box["result"] = run_pipeline(source, cfg, token, lambda f, m: events.put((f, m)))
            except Cancelled:
                box["error"] = "Cancelled. Temp files were removed."
            except Exception as exc:  # show the real message in the UI
                box["error"] = f"Error: {exc}"
            finally:
                events.put(None)

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        log: list = []
        try:
            while True:
                item = events.get()
                if item is None:
                    break
                frac, msg = item
                progress(frac, desc=msg)
                if not log or log[-1] != msg:
                    log.append(msg)
                yield "\n".join(log[-6:]), None
            if "error" in box:
                yield box["error"], None
            else:
                res = box["result"]
                files = [c["path"] for c in res["clips"]]
                lines = [f"Saved {len(files)} clips to: {res['output_dir']}", ""]
                for i, c in enumerate(res["clips"], 1):
                    lines.append(f"#{i}  {c['start']:.0f}s-{c['end']:.0f}s  score {c['score']}  "
                                 f"({', '.join(c['reasons']) or 'n/a'})")
                yield "\n".join(lines), files
        finally:
            if t.is_alive():          # browser tab closed mid-job
                token.cancel()
                t.join(timeout=15)
            _CURRENT["token"] = None
            _JOB_LOCK.release()

    def cancel_job():
        tok = _CURRENT["token"]
        if tok is None:
            return "Nothing is running."
        tok.cancel()
        return "Cancelling... (stops at the next safe point)"

    with gr.Blocks(title="Low-Spec Clipper") as demo:
        gr.Markdown("## Low-Spec AI Clipper\nCPU-only - faster-whisper INT8 + FFmpeg. "
                    "Use a **local file path** for big VODs (no extra copy on your SSD).")
        source = gr.Textbox(label="Video URL or local file path",
                            placeholder="https://youtu.be/...   or   D:\\vods\\stream.mp4")
        with gr.Row():
            model = gr.Radio(["tiny", "base"], value="base", label="Whisper model")
            lang = gr.Dropdown(["auto", "en", "sw"], value="auto", label="Language")
            height = gr.Dropdown([360, 480, 720], value=720, label="Max download height")
        with gr.Row():
            min_len = gr.Slider(15, 60, value=30, step=5, label="Min clip (s)")
            max_len = gr.Slider(30, 90, value=60, step=5, label="Max clip (s)")
            n = gr.Slider(1, 10, value=5, step=1, label="Number of clips")
        with gr.Row():
            reframe = gr.Radio(["crop", "blur", "pad"], value="crop", label="9:16 mode")
            face = gr.Checkbox(value=True, label="Face-aware crop (Haar)")
            subs = gr.Checkbox(value=True, label="Burn subtitles")
        with gr.Row():
            go = gr.Button("Generate clips", variant="primary")
            stop = gr.Button("Cancel", variant="stop")
        status = gr.Textbox(label="Status", lines=8, interactive=False)
        files = gr.File(label="Finished clips", file_count="multiple")

        go.click(run_job, [source, model, lang, min_len, max_len, n, reframe, face, subs, height],
                 [status, files])
        stop.click(cancel_job, None, status, concurrency_limit=None)
    demo.queue(default_concurrency_limit=1)
    return demo


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run_cli(args) -> int:
    cfg = Config(
        model_size=args.model, language=args.lang, num_clips=args.clips,
        min_clip_s=args.min_len, max_clip_s=args.max_len, reframe=args.reframe,
        face_detect=not args.no_face, burn_subs=not args.no_subs,
        max_source_height=args.height,
    )
    token = CancelToken()
    last = {"msg": ""}

    def show(frac, msg):
        if msg != last["msg"] or frac >= 1.0:
            print(f"[{frac * 100:5.1f}%] {msg}", flush=True)
            last["msg"] = msg

    try:
        res = run_pipeline(args.source, cfg, token, show)
    except KeyboardInterrupt:
        token.cancel()
        print("Interrupted.")
        return 130
    except Cancelled:
        print("Cancelled.")
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"\nDone -> {res['output_dir']}")
    for c in res["clips"]:
        print(f"  {Path(c['path']).name}  score={c['score']}  {', '.join(c['reasons'])}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Low-spec AI video clipper")
    ap.add_argument("source", nargs="?", help="URL or local file (omit to launch the web UI)")
    ap.add_argument("--model", choices=["tiny", "base"], default="base")
    ap.add_argument("--lang", default=None, help="e.g. en, sw (default: auto-detect)")
    ap.add_argument("--clips", type=int, default=5)
    ap.add_argument("--min-len", type=float, default=30)
    ap.add_argument("--max-len", type=float, default=60)
    ap.add_argument("--reframe", choices=["crop", "blur", "pad"], default="crop")
    ap.add_argument("--height", type=int, default=720, help="max download height")
    ap.add_argument("--no-face", action="store_true")
    ap.add_argument("--no-subs", action="store_true")
    args = ap.parse_args()

    if args.source:
        raise SystemExit(run_cli(args))

    out = Path("outputs").resolve()
    out.mkdir(exist_ok=True)
    build_ui().launch(server_name="127.0.0.1", server_port=7860,
                      inbrowser=True, allowed_paths=[str(out)])


if __name__ == "__main__":
    main()
