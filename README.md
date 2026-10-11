# Low-Spec AI Clipper

Local, CPU-only video clipper for a dual-core laptop (tested design target: i5-6200U, 8 GB RAM, no GPU).
Pipeline: yt-dlp -> mono 16 kHz WAV -> faster-whisper INT8 -> rule-based scoring -> one FFmpeg pass per clip (9:16 + burned subtitles).

```
video-clipper-lite/
├── main.py            # Gradio UI (progress + Cancel) and CLI
├── pipeline.py        # all the logic (threads, download, ASR, scoring, FFmpeg)
├── requirements.txt
└── README.md
```

## 1. Install (Windows 10/11 shown; Linux notes in brackets)

1. **Python 3.10-3.12** (64-bit). **FFmpeg** with libass (the "full" or "essentials" build from gyan.dev):
   `winget install Gyan.FFmpeg`  [Linux: `sudo apt install ffmpeg`]. Reopen the terminal, check `ffmpeg -version`.
2. Create a venv and install:
   ```
   python -m venv .venv
   .venv\Scripts\activate          [Linux: source .venv/bin/activate]
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. **PyTorch is NOT required.** faster-whisper uses CTranslate2, so skip torch and save ~1-2 GB of SSD.
   Only if you later add a torch-based tool, install the CPU build:
   `pip install torch --index-url https://download.pytorch.org/whl/cpu`
   (`pipeline.py` already caps torch to 2 threads if it is loaded).

## 2. Run

```
python main.py                       # web UI at http://127.0.0.1:7860
python main.py "https://youtu.be/ID" --model tiny --clips 4 --height 480   # CLI, lighter than the UI
python main.py "D:\vods\stream.mp4"  # local file: used in place, never copied
```
First run downloads the Whisper model (tiny ~75 MB, base ~145 MB) to the Hugging Face cache.
Clips + `clips.json` land in `outputs/<title>_<timestamp>/`. All temp files (source, WAV, SRTs) are deleted automatically, also on cancel/error.

## 3. Skylake / low-spec tuning

| Setting | Why |
|---|---|
| `CLIPPER_THREADS=2` (default) | 2 physical cores. Using all 4 logical threads causes thrashing, not speed. Set before launch if you want to change it. |
| `OMP/MKL/OPENBLAS_NUM_THREADS=2`, `OMP_WAIT_POLICY=PASSIVE` | Set in `pipeline.py` before numpy/ctranslate2 import (that is why `import pipeline` is first in `main.py`). |
| `compute_type="int8"`, `beam_size=1`, `temperature=0`, VAD on | Fastest faster-whisper settings on AVX2 CPUs. |
| Process priority | Whole run is below-normal priority so Windows stays usable. |
| `--height 480` / `preset superfast` | Fastest download + render. 720 looks sharper after the 9:16 crop but costs more CPU. |
| Power plan | Plug in and use "Best performance"; the 15 W U-series chip throttles hard on battery. |
| Close the browser tabs/Chrome | 8 GB goes quickly; the pipeline itself stays well under ~1.5 GB. |

Rough expectations (estimates, measure on your machine): `tiny` transcribes several times faster than real time, `base` roughly 1.5-3x; each 45 s clip renders in about its own duration.

## 4. Notes

- **Language**: tiny/base are weak on Swahili/Sheng and code-switching. If transcripts look wrong, force `--lang sw` or `en`, or try `small` (set `model_size="small"` in `Config`; slower, ~2x RAM).
- **Scoring** is heuristic (questions, emotion words, laughter, numbers, speech pace, loudness, strong opening, clean ending). Edit `EMOTION_WORDS` in `pipeline.py` with your streamers' catchphrases.
- **Face crop** uses one static Haar-cascade crop per clip (no tracking). It suits single-speaker streams; for multi-person scenes use `blur` mode.
- yt-dlp breaks when sites change: `pip install -U yt-dlp` first when downloads fail.
