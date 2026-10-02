from response_analysis import Qwen 
run = True;

def main():
    print("Main starts")
    # Create the reponse analysis model
    analysis_model = Qwen("Qwen/Qwen3-8B", "persona1.txt")
    print("Model is loaded")

    while(run == True):
        user_response = input("Enter your response: ");

        if(user_response == "q"):
            quit()

        response = analysis_model.generate_response(user_response)
        print(response)

if __name__ == "__main__":
    main()
    