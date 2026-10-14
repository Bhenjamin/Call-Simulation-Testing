import time
from response_analysis import ResponseAnalysis 
from speech_to_text import SpeechTranscription
from text_to_speech import SpeechOutput 

# Speech to Text model used: https://github.com/SYSTRAN/faster-whisper
STT_MODEL = "large-v3"

# Model used to generate caller response: https://ollama.com/library/qwen3:4b-instruct
RESP_MODEL = "qwen3:4b-instruct"

# Text to Speech model used: https://github.com/QwenLM/Qwen3-TTS
TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

def main():
    print("Main starts")

    start_time = time.time()

    # Intialises the reponse analysis model
    analysis_model = ResponseAnalysis(RESP_MODEL, "persona1.txt")
    print("Analysis Model is loaded")

    # Initialises TTS model
    speech_recognition_model = SpeechTranscription(STT_MODEL)
    print("Speech Recognition Model is loaded")

    # Intialises STT model
    speech_model = SpeechOutput(TTS_MODEL)
    print("Speech Model is loaded")

    end_time = time.time()
    print(f"Model load time: {end_time - start_time}")

    while(True):
        # Transcribe the user's response
        user_response = speech_recognition_model.transcribe_audio()

        if(user_response == "quit"):
            break

        # Create the caller's response
        response = analysis_model.generate_response(user_response)
        print(response)

        # Generate and output the caller's reponse as audio 
        speech_model.generate_audio(response)

if __name__ == "__main__":
    main()
    