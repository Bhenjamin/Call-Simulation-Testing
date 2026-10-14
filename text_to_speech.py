import torch
import soundfile as sf
import winsound
from qwen_tts import Qwen3TTSModel

# Voice cloning audio reference
REF_AUDIO = "./voice_references/New_Zealand_Accent_Slang_onevoice.wav"

# Text that matches what the audio says - must be word for word
REF_TEXT = "I think we talk alright eh, I think their slow, ehahahaha, rather than we talk to fast, I think they think to slow"

# Location of generated audio wave file 
OUTPUT_FILE = "./caller_audio_output/output_voice_clone.wav"

class SpeechOutput:
    def __init__(self, model_name):
        self.model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2", - Not compatible with GTX GPUs
            attn_implementation="sdpa",
        )

        self.cloned_voice = self.model.create_voice_clone_prompt(
            ref_audio=REF_AUDIO,
            ref_text=REF_TEXT,
            x_vector_only_mode=False,
        )

    def generate_audio(self, voice_line: str):
        # TODO: Improve the voice - Temp voice cloning video - will aim to improve in future

        wavs, sr = self.model.generate_voice_clone(
            # This is the output text that you want to be spoken in the cloned voice, you can change this to whatever you want
            text=voice_line,
            language="English",
            voice_clone_prompt=self.cloned_voice,
        )

        # Writes the audio to file
        sf.write(OUTPUT_FILE, wavs[0], sr)

        # Plays audio back to user
        winsound.PlaySound(OUTPUT_FILE, winsound.SND_FILENAME)