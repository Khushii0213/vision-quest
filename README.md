# Vision Quest

A timed object-matching arcade game with a computer-vision laboratory attached.

## 🎮 Live Demo

**[🚀 Play Vision Quest](https://vision-quest-qxarfrhk9tafxwfvoed2tz.streamlit.app/)**

Try the interactive object-matching arcade game and explore how a pretrained MobileNetV3-Small model responds to image distortions.

> **Note:** The AI Lab and Webcam Lab demonstrate image classification using a pretrained ImageNet-1K model. The model's predictions are for exploration and are not used to calculate game scores.

---
Each round names an object (for example "cat"). You pick the one tile that shows it from a mosaic of look-alikes that are rotated, darkened, blurred or noisy. The AI Lab and Webcam Lab run a real pretrained neural network (MobileNetV3-Small, ImageNet-1K) so you can see how it reacts to the same distortions.

## Features

| Screen | What it does |
|---|---|
| **Arcade** | Easy, Normal and Hard runs with a live countdown, speed bonus, streaks, lives, limited skips, round reveal showing the true label of every tile, game-over summary, best score saved locally. |
| **AI Lab** | Pick a CIFAR-10 image or upload your own. Adjust brightness, rotation, blur, contrast, noise and gamma. Compare top-5 predictions before and after, with rank/confidence changes marked. Reset button, technical notes on preprocessing, inference, softmax and confidence. |
| **Webcam Lab** | Take a browser photo (or upload one) and classify it. Clear messages when the camera is missing or blocked. |
| **Results** | Last run, best score and streak, totals, average answer time, score history, reset. |

## Quick start (macOS)

Requires Python 3.11 or 3.12 (`brew install python@3.12` or download from python.org) and an internet connection **for the first launch only**.

```bash
cd vision-quest
python3.12 -m venv .venv            # or python3.11
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/preflight.py --download   # one-time: CIFAR-10 (~170 MB) + model weights (~10 MB)
streamlit run app.py
```

The app opens at <http://localhost:8501>. Next time only `source .venv/bin/activate` and `streamlit run app.py` are needed, and it works offline.
Shortcut for all of the above: `bash scripts/setup_mac.sh`.

If you skip the preflight download, the app shows a first-run screen with a **Download dataset** button, and the labs download the model when first opened.

## How to use

1. **Arcade**: choose a difficulty, press *Start run*. Read the target name, find it in the grid, press the matching *Pick* button (tiles are labelled A1, A2, ...). After each round the true label of every tile is shown. *Skip round* is limited. Leaving the arcade abandons the run (timers cannot run in the background), and abandoned runs are not saved.
2. **AI Lab**: choose a class and sample (or upload), move the sliders, read the two top-5 lists and the "What changed" table. *Reset experiment* restores neutral settings.
3. **Webcam Lab**: allow the camera, press the camera button to take a photo, read the predictions. *Capture again* clears the photo.
4. **Results**: review history; the reset needs a confirmation checkbox.

### Scoring rules

| Part | Value |
|---|---|
| Base for a correct pick | 100 |
| Speed bonus | `floor(50 x time_left / time_limit)` |
| Streak bonus | `20 x min(streak_before, 5)` |
| Difficulty multiplier | Easy x1, Normal x1.5, Hard x2 (integer percent arithmetic) |
| Correct total | `(100 + speed + streak) x multiplier` |
| Wrong, timed out, skipped | 0 points, streak resets |
| Lives | Wrong and timed-out rounds cost one life; skips do not |

| | Easy | Normal | Hard |
|---|---|---|---|
| Rounds / lives / skips | 10 / 5 / 3 | 12 / 3 / 2 | 15 / 2 / 1 |
| Tiles | 4, then 6 | 6, then 9 | 8, then 12 (grid grows from round 6) |
| Time per round | 20 s down to 12 s | 15 s down to 8 s | 12 s down to 5 s |
| Distortion strength | 0.15 up to 0.50 | 0.30 up to 0.75 | 0.50 up to 1.00 |

Exactly one tile per round shows the target class; every tile is a different dataset image.

## Dataset and labels

The game uses the **CIFAR-10 test split** (10,000 labelled 32 x 32 images, 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck), downloaded with `torchvision` from the University of Toronto site and cached in `data/`.

**Game labels are CIFAR-10 ground truth.** Scoring never uses the neural network. The network predicts **ImageNet-1K** classes (1,000 labels such as "tabby cat"), a different label set, so its output is shown for exploration and is never compared with the CIFAR label or called "correct".

Consequence of the choice: CIFAR-10 images are tiny. They are enlarged (Lanczos) to 192 px for tiles and 256 px for the AI Lab; enlarging adds no detail, so tiles look soft and the ImageNet model often struggles on them. We chose CIFAR-10 because it is small, standard, reliably downloadable, and has trustworthy ground-truth labels, which the game needs.

## AI pipeline

`PIL RGB image -> resize 256 (bicubic) -> centre-crop 224 -> scale to 0-1 -> normalise (ImageNet mean/std) -> MobileNetV3-Small (eval mode, torch.inference_mode) -> softmax -> top-5`.

- Weights: `MobileNet_V3_Small_Weights.IMAGENET1K_V1`, preprocessing taken from the weights object itself.
- Loaded once per server process (`st.cache_resource`) and reused. Runs on CPU.
- No fallback: if the model cannot load or run, the screen says so and shows no predictions. Nothing is simulated.
- Inference time shown in the labs is measured live on your machine.

### Concepts

- **Image classification**: one label (or ranked list) for the whole image. This is what the app does.
- **Object detection**: locates objects with boxes. Not done here.
- **Data augmentation**: transforming images, normally to enlarge training data; here used to probe the model.
- **Transfer learning**: reusing a network trained on one task for another, usually with fine-tuning. We use the pretrained network as-is.
- **Inference**: running a trained model on new input.
- **Confidence vs accuracy**: confidence is the softmax probability the model reports; accuracy is agreement with ground truth over many examples. A single confidence value proves neither.

### Augmentations (Pillow + NumPy)

Applied in this order: rotation (reflect-filled corners) -> contrast -> brightness -> gamma -> blur -> noise.

| Control | Range | Neutral |
|---|---|---|
| Brightness factor | 0.2 to 2.0 | 1.0 |
| Rotation | -180 to 180 degrees | 0 |
| Blur (Gaussian radius) | 0 to 6 px | 0 |
| Contrast factor | 0.2 to 2.0 | 1.0 |
| Noise (Gaussian std-dev, 0-255 scale) | 0 to 60 | 0 |
| Gamma (`out = in^gamma`, above 1 darkens: low light) | 0.4 to 3.0 | 1.0 |

Out-of-range values raise an error instead of being silently clipped.

## Project layout

```
vision-quest/
├── app.py                  entry point and screen router
├── requirements.txt        runtime dependencies
├── requirements-dev.txt    + pytest
├── assets/                 styles.css, logo.svg, favicon.png
├── .streamlit/config.toml  theme and server settings
├── src/
│   ├── config.py           paths and constants (all relative to the project)
│   ├── dataset.py          CIFAR-10 loader (ground-truth labels)
│   ├── model.py            MobileNetV3-Small classifier
│   ├── augmentation.py     transforms and random tile parameters
│   ├── game.py             game engine / state machine (no UI code)
│   ├── scoring.py          scoring rules
│   ├── storage.py          local JSON persistence
│   ├── compare.py          original-vs-transformed prediction comparison
│   ├── experiment.py       prediction-stability experiment
│   ├── imageio.py          safe image loading
│   ├── services.py         cached dataset/model/storage accessors
│   └── ui/                 Streamlit screens: home, arcade, lab, webcam, results
├── tests/                  pytest suite
├── scripts/                preflight.py, run_experiment.py, setup_mac.sh
└── docs/                   PROJECT_REPORT.md, PRESENTATION.md, TEAM_DIVISION.md
```

Saved progress lives in `data/vision_quest.json` (best score, totals, last 50 runs). Set `VISION_QUEST_DATA_DIR` to change the location. `data/` is never packaged or committed.

## Testing

```bash
source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest -q                       # unit + headless UI tests (use fakes, no downloads)
python scripts/preflight.py --smoke       # ONE real inference on a real CIFAR-10 image (needs downloads)
python scripts/run_experiment.py --images 200   # real stability measurements, writes data/experiments/stability.csv
```

The pytest suite covers scoring, game state transitions (double-scoring, late clicks, timeouts, skips, lives, run completion), augmentation, storage (including corrupt files), image validation, prediction comparison, requirements-versus-imports, and headless runs of every screen. It uses test-only fakes for the dataset and classifier (`tests/fakes.py`), so it does **not** validate real-model accuracy; the `--smoke` and experiment scripts do that on your machine.

## Troubleshooting

| Problem | Fix |
|---|---|
| `pip install torch` fails | Use Python 3.11 or 3.12 in a fresh `.venv`. Check `python --version`. |
| Intel Mac | Current PyTorch has no Intel-Mac wheels. Use `pip install "numpy<2" "torch==2.2.2" "torchvision==0.17.2" pillow streamlit`. |
| `CERTIFICATE_VERIFY_FAILED` on download | python.org installs: run `/Applications/Python 3.12/Install Certificates.command`, then retry. |
| Download fails or is slow | Re-run `python scripts/preflight.py --download`. If a partial file exists, delete `data/datasets` and retry. |
| Must work offline at the demo | Run the preflight download beforehand at home; afterwards everything is cached in `data/`. |
| Camera does not start | Allow it in the browser prompt and in System Settings, Privacy & Security, Camera. Use `http://localhost:8501`. Quit Zoom/FaceTime. Or use *Upload an image*. |
| Port 8501 busy | `streamlit run app.py --server.port 8502` |
| Fonts look different offline | Fonts load from Google Fonts when online and fall back to system fonts offline. Layout still works. |
| Progress file damaged | It is renamed to `vision_quest.corrupt` and a fresh one starts. |

## Known limitations

- Real-browser behaviour (camera permission prompts, the 0.25 s timer redraw on a slow laptop, visual layout) could not be checked in the automated environment and should be verified on the target Mac before the demo.
- The countdown is a Streamlit fragment refreshing four times per second, so the display can lag slightly on slow machines. The deadline itself is a monotonic-clock timestamp, so scoring and expiry are exact regardless of redraw speed.
- CIFAR-10 is low resolution; the ImageNet model's predictions on it are weak and should not be read as a quality statement about MobileNetV3.
- The AI model is not used in game scoring. It does not detect objects, only classifies whole images.
- Single-player, local only. Leaving the arcade mid-run abandons the run.

## References

- Howard et al., *Searching for MobileNetV3*, ICCV 2019. <https://arxiv.org/abs/1905.02244>
- torchvision `mobilenet_v3_small` and `MobileNet_V3_Small_Weights.IMAGENET1K_V1`. <https://pytorch.org/vision/stable/models/mobilenetv3.html>
- Russakovsky et al., *ImageNet Large Scale Visual Recognition Challenge*, IJCV 2015.
- Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009 (CIFAR-10). <https://www.cs.toronto.edu/~kriz/cifar.html>
- Streamlit documentation. <https://docs.streamlit.io>


