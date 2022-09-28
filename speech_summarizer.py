import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# INPUT_AUDIO_FILE=r"input/speech.wav"
# OUTPUT_SUMMARISED_FILE=r"output/summarised_text.txt"
OUTPUT_TRANSCRIPT_FILE=r"output/transcript.txt"

DEEPSPEECH_MODEL = r"models/deepspeech-0.9.3-models.tflite"
PYTORCH_MODEL_URL=r"https://cdn-lfs.huggingface.co/facebook/bart-large-cnn/2ac2745c02ac987d82c78a14b426de58d5e4178ae8039ba1c6881eccff3e82f1"
PYTORCH_MODEL_FILE=r"models/bart-large-cnn/pytorch_model.bin"
PYTORCH_MODEL_DIR = r"models/bart-large-cnn"

def run_app(INPUT_AUDIO_FILE=r"input/speech.wav", OUTPUT_SUMMARISED_FILE=r"output/summarised_text.txt"):
    if not os.path.exists(PYTORCH_MODEL_FILE):
        #Download models for summarisation if doesn't exist
        print("Downloading pretrained model for summarisation ...")
        print("It might take from 2 to 10 minutes based on bandwidth!")
        CMD_TO_EXEC = f"curl -o {PYTORCH_MODEL_FILE} {PYTORCH_MODEL_URL}"
        os.system(CMD_TO_EXEC)
    else:
        print("Pretrained model found.")

    print("Transcribing ...")
    #Use deepspeech model to convert a speech file (e.g., speech.wav) into text file transcript.txt
    CMD_TO_EXEC = f"deepspeech --model {DEEPSPEECH_MODEL} --audio {INPUT_AUDIO_FILE} > {OUTPUT_TRANSCRIPT_FILE}"
    os.system(CMD_TO_EXEC)

    #Load generated transcript into memory
    transcribed_text = str()
    with open(OUTPUT_TRANSCRIPT_FILE) as file:
        transcribed_text = file.read().strip()

    print("Summarising ...")
    #Summarize transcribed speech
    tokenizer = AutoTokenizer.from_pretrained(PYTORCH_MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(PYTORCH_MODEL_DIR)

    input_ids = tokenizer(f"summarize: {transcribed_text}", return_tensors='pt').input_ids
    outputs = model.generate(input_ids)
    with open(OUTPUT_SUMMARISED_FILE, 'w') as file:
        file.write(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("----------------------------")
    print("Summarised output: ", tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("----------------------------")
    print(f"Output is written to the file: {OUTPUT_SUMMARISED_FILE}")
