"""Run: python main.py  ->  http://127.0.0.1:7860"""
import gradio as gr
from pathlib import Path
from pipeline import JobConfig, JobRunner

runner = JobRunner()

def start_job(url, model_size, clip_len, max_clips, vertical, subtitles, face_track):
    if not url.strip():
        return "Paste a URL first.", gr.update(visible=False)
    cfg = JobConfig(
        url=url.strip(), out_dir=Path("clips"), model_size=model_size,
        clip_min=max(15, int(clip_len) - 15), clip_max=int(clip_len),
        max_clips=int(max_clips), vertical=bool(vertical),
        burn_subtitles=bool(subtitles), face_track=bool(face_track))
    try:
        runner.start(cfg); return "Job started…", gr.update(visible=True)
    except RuntimeError as e:
        return str(e), gr.update(visible=False)

def poll():
    s = runner.state
    bar = gr.update(value=s.progress)
    if s.stage == "done":
        vids = [str(p) for p in sorted(Path("clips").glob("clip_*.mp4"))]
        return bar, f"**{s.msg}**", gr.update(visible=True, value=vids)
    if s.stage == "error":
        return bar, f"❌ {s.error}", gr.update(visible=False)
    if s.stage == "cancelled":
        return bar, "Cancelled.", gr.update(visible=False)
    return bar, f"{s.stage}: {s.msg}", gr.update(visible=False)

def cancel_job():
    runner.cancel()
    return "Cancelling… (current step will abort)"

with gr.Blocks(title="Local AI Clipper (low-spec)") as demo:
    gr.Markdown("# 🎬 Local AI Clipper\nCPU-only • INT8 Whisper • FFmpeg streaming")
    with gr.Row():
        url = gr.Textbox(label="Video URL (YouTube etc.)", scale=3)
        model = gr.Dropdown(["tiny", "base"], value="base",
                            label="Whisper model (tiny = fastest)")
    with gr.Row():
        clip_len = gr.Slider(30, 60, value=45, step=15, label="Clip length (s)")
        max_clips = gr.Slider(1, 10, value=5, step=1, label="Max clips")
    with gr.Row():
        vertical = gr.Checkbox(value=True, label="9:16 vertical reframe")
        subs = gr.Checkbox(value=True, label="Burn subtitles")
        face = gr.Checkbox(value=False, label="Face tracking (Haar, slower)")
    btn = gr.Button("Generate Clips", variant="primary")
    cancel = gr.Button("Cancel", visible=False)
    bar = gr.Progress(); status = gr.Markdown("Idle.")
    out = gr.Files(label="Clips", visible=False)

    btn.click(start_job, [url, model, clip_len, max_clips, vertical, subs, face],
              [status, cancel])
    cancel.click(cancel_job, None, status)
    gr.Timer(1.0).tick(poll, None, [bar, status, out])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", share=False)
