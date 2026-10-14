import time
from response_analysis import ResponseAnalysis 
from speech_to_text import SpeechTranscription

def main():
    print("Main starts")

    # Create the reponse analysis model
    analysis_model = ResponseAnalysis("qwen3:4b-instruct", "persona1.txt")

    # Transcribes user voice input to text
    speech_recognition_model = SpeechTranscription()
    
    print("Model is loaded")

    while(True):

        user_response = speech_recognition_model.transcribe_audio()

        if(user_response == "quit"):
            break

        response = analysis_model.generate_response(user_response)
        print(response)

if __name__ == "__main__":
    main()
    