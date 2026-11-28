import torch
import soundfile as sf
import winsound
import json
from qwen_tts import Qwen3TTSModel

# Voice cloning audio reference
# REF_AUDIO = "./voice_references/New_Zealand_Accent_Slang_onevoice.wav"
REF_AUDIO = "./voice_references/"

# Text that matches what the audio says - must be word for word
# REF_TEXT = "I think we talk alright eh, I think their slow, ehahahaha, rather than we talk to fast, I think they think to slow"
REF_TEXT = "Hello my name is Benjamin this is a test for voice cloning, hopefully it can pick up on my New Zealand accent"

# Location of generated audio wave file 
OUTPUT_FILE = "./caller_audio_output/output_voice_clone.wav"

class SpeechOutput:
    def __init__(self, model_name, voice_option):
        self.model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map="cuda:0",
            dtype=torch.float32,
            # attn_implementation="flash_attention_2", - Not compatible with GTX GPUs - test again later
            attn_implementation="sdpa",
        )

        # Voice data from JSON file
        with open("voice_references/voice_metadata.json", "r", encoding="utf-8") as file:
            self.voice_data = json.load(file)


        # Chosen voice currently based on cmd line input
        if(voice_option is None):
            raise ValueError("No emotion was selected for the caller's voiceline") 

        match voice_option:
            case 1:
                self.voice_option = "priya"
            case 2:
                self.voice_option = "grace"
            case 3:
                self.voice_option = "marcus"
            case 4:
                self.voice_option = "hannah"

        print(self.voice_option)

        # Preloads all the voices different emotional tones
        self.angry_voice = self.model.create_voice_clone_prompt(
            ref_audio=self.voice_data[self.voice_option+"_angry"]["filepath"],
            ref_text=self.voice_data[self.voice_option+"_angry"]["voice_line"],
            x_vector_only_mode=False
        )

        self.happy_voice= self.model.create_voice_clone_prompt(
            ref_audio=self.voice_data[self.voice_option+"_happy"]["filepath"],
            ref_text=self.voice_data[self.voice_option+"_happy"]["voice_line"],
            x_vector_only_mode=False
        )

        self.neutral_voice = self.model.create_voice_clone_prompt(
            ref_audio=self.voice_data[self.voice_option+"_neutral"]["filepath"],
            ref_text=self.voice_data[self.voice_option+"_neutral"]["voice_line"],
            x_vector_only_mode=False
        )

        self.sad_voice= self.model.create_voice_clone_prompt(
            ref_audio=self.voice_data[self.voice_option+"_sad"]["filepath"],
            ref_text=self.voice_data[self.voice_option+"_sad"]["voice_line"],
            x_vector_only_mode=False
        )

    def generate_audio(self, voice_line: str, voice_emotion: str):
        # TODO: Improve the voice - Temp voice cloning video - will aim to improve in future

        current_voice = ""
        match voice_emotion:
            case "a":
                current_voice = self.angry_voice
            case "h":
                current_voice = self.happy_voice
            case "n":
                current_voice = self.neutral_voice
            case "s":
                current_voice = self.sad_voice

        if(current_voice == ""):
            raise ValueError("Voice was not selected, when trying to generate audio")

        wavs, sr = self.model.generate_voice_clone(
            # What the AI will be saying
            text=voice_line,

            language="English",
            # Selected voice to use and emotion
            voice_clone_prompt=current_voice
        )

        # Writes the audio to file
        sf.write(OUTPUT_FILE, wavs[0], sr)

        # Plays audio back to user
        winsound.PlaySound(OUTPUT_FILE, winsound.SND_FILENAME)