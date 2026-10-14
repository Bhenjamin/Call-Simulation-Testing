import torch
import time
import soundfile as sf
import winsound
from qwen_tts import Qwen3TTSModel

# Voice cloning audio reference
# REF_AUDIO = "./voice_references/New_Zealand_Accent_Slang_onevoice.wav"
REF_AUDIO = "./voice_references/test1.wav"

# Text that matches what the audio says - must be word for word
# REF_TEXT = "I think we talk alright eh, I think their slow, ehahahaha, rather than we talk to fast, I think they think to slow"
REF_TEXT = "Hello my name is Benjamin this is a test for voice cloning, hopefully it can pick up on my New Zealand accent"

# Location of generated audio wave file 
OUTPUT_FILE = "./caller_audio_output/output_voice_clone.wav"

class SpeechOutput:
    def __init__(self, model_name):
        self.model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map="cuda:0",
            dtype=torch.float32,
            # attn_implementation="flash_attention_2", - Not compatible with GTX GPUs
            attn_implementation="sdpa",
        )

        self.cloned_voice = self.model.create_voice_clone_prompt(
            ref_audio=REF_AUDIO,
            ref_text=REF_TEXT,
            x_vector_only_mode=False,
        )

    def generate_audio(self, voice_line: str):
        start = time.perf_counter()

        # Generate speech
        wavs, sr = self.model.generate_voice_clone(
            text=voice_line,
            language="English",
            voice_clone_prompt=self.cloned_voice,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        generated = time.perf_counter()
        print(f"TTS generation: {generated - start:.2f} seconds")

        # Save audio
        sf.write(OUTPUT_FILE, wavs[0], sr)

        written = time.perf_counter()
        print(f"File writing: {written - generated:.2f} seconds")

        # Play audio
        winsound.PlaySound(OUTPUT_FILE, winsound.SND_FILENAME)

        played = time.perf_counter()
        print(f"Playback: {played - written:.2f} seconds")
        print(f"Total TTS time: {played - start:.2f} seconds")


    # def generate_audio(self, voice_line: str):
    #     # TODO: Improve the voice - Temp voice cloning video - will aim to improve in future

    #     wavs, sr = self.model.generate_voice_clone(
    #         # This is the output text that you want to be spoken in the cloned voice, you can change this to whatever you want
    #         text=voice_line,
    #         language="English",
    #         voice_clone_prompt=self.cloned_voice,
    #     )

    #     # Writes the audio to file
    #     sf.write(OUTPUT_FILE, wavs[0], sr)

    #     # Plays audio back to user
    #     winsound.PlaySound(OUTPUT_FILE, winsound.SND_FILENAME)