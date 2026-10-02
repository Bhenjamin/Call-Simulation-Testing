## Qwen3-TTS Setup
Requirements
- NVIDIA GPU with CUDA support
- NVIDIA drivers installed
- Miniconda
- Python 3.12

***

#### Setup
> Note: Also ensure you have mini conda or conda installed

###### TTS Setup
**Create and activate the environment:**
```
conda create -n call_simulation_prototype python=3.12
conda activate call_simulation_prototype
```

**Install CUDA-enabled PyTorch: will change depending on the GPU - This one is for my GPU**
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
```
True
NVIDIA ...
```

***

###### Analysis Model Setup
pip install -U transformers

Run
python test_qwen.py