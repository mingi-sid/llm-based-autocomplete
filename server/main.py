# Simple server to handle requests from the client
# and send back the appropriate response
from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, TextStreamer
from peft import LoraConfig, TaskType, PeftModel
from peft import get_peft_config, get_peft_model

max_length = 400

model_id = "heegyu/kogpt-j-base"
device = 'cuda'

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="models")
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir="models")

_ = model.eval()
_ = model.to(device)

# load peft_config from json format
model = PeftModel.from_pretrained(model, 'models/heegyu_kogpt-j-base_20230802142007')

app = Flask(__name__)
CORS(app)

# Function to perform autocomplete based on the input text
def perform_autocomplete(input_text):
    # Auto-complete logic goes here
    print(f'Starting generating. Input length: {len(input_text)}')
    input_text_truncated = input_text[-max_length:]
    with torch.no_grad():
        tokens = tokenizer(input_text_truncated, return_tensors='pt').to(device='cuda')
        #streamer = TextStreamer(tokenizer)
        gen_tokens = model.generate(
            input_ids=tokens['input_ids'],
            attention_mask=tokens['attention_mask'],
            #streamer=streamer,
            do_sample=True,
            temperature=0.9,
            max_new_tokens=100,
            top_k=10,
            repetition_penalty=1.0,
            eos_token_id=[tokenizer.eos_token_id,
                          tokenizer.convert_tokens_to_ids('\n'),
                          tokenizer.convert_tokens_to_ids('<s>')],
        )
        generated = tokenizer.batch_decode(gen_tokens)
    generated_text = generated[0]
    newly_generated_text = generated_text[len(input_text_truncated):]
    print(f'Newly generated text length: {len(newly_generated_text)}')
    return newly_generated_text

# Endpoint to handle autocomplete requests
@app.route('/autocomplete', methods=['POST'])
def autocomplete():
    data = request.get_json()
    input_text = data.get('inputText', '')

    if not input_text:
        response = jsonify({'autoCompletedText': ''})
    else:
        # Call the function to perform autocomplete based on the input text
        auto_completed_text = perform_autocomplete(input_text)
        response = jsonify({'autoCompletedText': auto_completed_text})

    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Content-Type', 'application/json')
    response.headers.add('Accept', 'application/json')
    return response

if __name__ == '__main__':
    app.run(debug=True)