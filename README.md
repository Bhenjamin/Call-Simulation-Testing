## Qwen3-TTS Setup
Requirements
- NVIDIA GPU with CUDA support
- NVIDIA drivers installed
- Miniconda
- Python 3.12

***

#### Setup

**Create and activate the environment:**
```
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
```

**Install CUDA-enabled PyTorch:**
```
pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

**Install Qwen3-TTS:**
```
pip install -U qwen-tts
```

**Verify GPU**
```
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Should output:

True
NVIDIA ...

Run
python test_qwen.py


The first run will automatically download the Qwen3-TTS model.

Use a .wav file for the voice reference.
