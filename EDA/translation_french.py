import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import pandas as pd
from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer

# Load the model, tokenizer
def load_model():
    model_name = 'Helsinki-NLP/opus-mt-fr-en'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to('cuda')
    return tokenizer, model

tokenizer, model = load_model()

def translate(text, tokenizer, model):
    inputs = tokenizer.encode(text, return_tensors="pt").to('cuda')
    outputs = model.generate(inputs, num_beams=4, max_length=512, early_stopping=True)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

def detect_language(text):
    try:
        return detect(text)
    except:
        return "error detecting language"

def translate_if_french(description, tokenizer, model, words_to_preserve):
    lang = detect_language(description)
    if lang == 'fr':
        translated_text = " ".join(
            translate(word, tokenizer, model) if word.lower() not in words_to_preserve else word
            for word in description.split()
        )
        return translated_text
    else:
        return description

input_csv_file = '/home/paa63/bd-2/Project/processed_selected_col_without_lang.csv'

df = pd.read_csv(input_csv_file, engine="python")

words_to_preserve = {"pierre", "poilievre", "Pierre", "Poilievre"}

for index, row in df.iterrows():
    df.at[index, 'Comment'] = translate_if_french(row['Comment'], tokenizer, model, words_to_preserve)

df.to_csv('/home/paa63/bd-2/Project/translations_data.csv', index=False)