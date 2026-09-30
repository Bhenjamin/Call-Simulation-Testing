import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",
    attn_implementation="sdpa",
)

# Added the path for .wav audio files here - this is the reference audio that will be used for voice cloning
ref_audio = "./audio-tests/New_Zealand_Accent_Slang_onevoice.wav"
# Here you have to write out word for word what is said in the audio file
ref_text  = "I think we talk alright eh, I think their slow, ehahahaha, rather than we talk to fast, I think they think to slow"

wavs, sr = model.generate_voice_clone(
    # This is the output text that you want to be spoken in the cloned voice, you can change this to whatever you want
    text="Keyora bro, I'm Ta nae and I'm calling in because I have an issue with my case, I have heard nothing back in weeks",
    language="English",
    ref_audio=ref_audio,
    ref_text=ref_text,
)
sf.write("output_voice_clone.wav", wavs[0], sr)