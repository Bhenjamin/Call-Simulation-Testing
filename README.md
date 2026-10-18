# Call Simulation Prototype

An AI caller for call simulation. The pipeline is:

1. Speech to text: faster-whisper
2. Caller response: Ollama running `qwen3:4b-instruct`
3. Text to speech with voice cloning: Qwen3-TTS

Windows only (audio playback uses `winsound`).

***

## Requirements

- Windows
- NVIDIA GPU with CUDA support and up-to-date drivers
- Miniconda
- Python 3.12
- Ollama (the app, installed separately from the Python packages)

***

## Quick setup

Run `setup.bat` from the repo folder. It does steps 1 to 6 below. If your GPU needs a different CUDA build of PyTorch, change `TORCH_INDEX` at the top of the file first.

## Setup (manual steps)

### 1. Create and activate the environment

```
conda create -n call_simulation_prototype python=3.12
conda activate call_simulation_prototype
```

### 2. Install PyTorch (depends on the GPU)

Pick the CUDA build that matches your GPU and driver at https://pytorch.org/get-started/locally/. This one is for the GTX 1650 SUPER development PC:

```
pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

Do this before step 3. Otherwise the next step can install a CPU-only PyTorch.

### 3. Install the other packages

```
pip install -r requirements.txt
```

Do not run `pip install -U transformers`. `qwen-tts` needs a specific older version (4.57.3), and upgrading it breaks the text to speech.

### 4. Verify the GPU

```
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Should output:

```
True
NVIDIA ...
```

### 5. Install Ollama and the response model

```
winget install Ollama.Ollama
```

Open a new terminal so the `ollama` command is found, then:

```
ollama pull qwen3:4b-instruct
ollama list
```

`pip install ollama` (already in requirements.txt) only installs the Python client. The Ollama app is the part that runs the model.

### 6. Create the folders and files that are not in the repo

```
mkdir user_audio
mkdir caller_audio_output
```

- Copy the voice cloning reference `.wav` into `voice_references/` (the path is set in `text_to_speech.py`).
- Check `personas/persona1.txt` exists.

***

## Run

```
conda activate call_simulation_prototype
python main.py
```

***

## Settings that depend on the machine

| Setting | Where | Notes |
|---|---|---|
| PyTorch CUDA build | Setup step 2 | Match the GPU and driver |
| Text to speech precision (`dtype`) | `text_to_speech.py` | `float32` is used on the GTX 1650 SUPER (`bfloat16` was slow and `float16` crashed). RTX 30-series and newer support `bfloat16`, so it is worth trying there |
| Text to speech model | `TTS_MODEL` in `main.py` | Qwen3-TTS Base models |
| Speech to text model and device | `STT_MODEL` in `main.py`, `speech_to_text.py` | On the 4 GB GPU, `large-v3` competed for memory. A smaller model on the CPU was faster. A larger GPU can run it on the GPU |
| Response model | `RESP_MODEL` in `main.py` | Any Ollama model |

Close other GPU-heavy programs before testing speed. A screen recorder made runs several times slower.

***

## Warnings that are safe to ignore

- `flash-attn is not installed`: not needed. It is not compatible with GTX GPUs, so the code uses `sdpa`.
- `SoX could not be found`: only a warning.

***

## Troubleshooting

| Problem | Fix |
|---|---|
| `ollama` is not recognized | Open a new terminal. If it still fails, the Ollama app is not installed (step 5) |
| `unexpected keyword argument 'metadata_errors'` | `pip install "av<19"` |
| Slow generation | Run `nvidia-smi` and `ollama ps` while it runs. The model should say 100% GPU |
| File not found for `user_audio` or `caller_audio_output` | Create the folders (step 6) |