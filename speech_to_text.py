import speech_recognition as sr
from faster_whisper import WhisperModel

AUDIO_FILEPATH = "./user_audio/audio_input.wav" 

class SpeechTranscription:

    def __init__ (self, model_name: str):
        # Run on GPU with FP16
        self.model = WhisperModel(model_name, device="cuda", compute_type="float16")
        # Class that detects and records audio
        self.recogniser = sr.Recognizer()


    def transcribe_audio(self)->str:
        # Record audio
        with sr.Microphone() as source:
            print("User speak now: ")
            audio = self.recogniser.listen(source)

        print("User has stopped speaking")

        # Write data to an wav audio file
        with open(AUDIO_FILEPATH, "wb") as f:
            f.write(audio.get_wav_data())
        

        segments, info = self.model.transcribe(AUDIO_FILEPATH, beam_size=5)

        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

        # Join each word together
        transcribed_audio = ""
        for segment in segments:
            transcribed_audio.join(segment.text)

        return transcribed_audio