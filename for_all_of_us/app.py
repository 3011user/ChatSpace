
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM
app = Flask(__name__)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2')

# Set padding token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id


def generate_response(input_text, model, tokenizer):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=200,
        num_return_sequences=1,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id  
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if response.lower().startswith(input_text.lower()):
        response = response[len(input_text):].strip()   

     # Additional check to remove partial repetition
    if response.lower().startswith(input_text.lower().split('?')[0].strip()):
        response = response[len(input_text):].strip() 
    
    # Ensure the response ends with a period or full sentence
    if not response.endswith(('.', '!', '?')):
        response = response.rsplit('.', 1)[0] + '.'

    return response
    

"""
def generate_response(input_text, model, tokenizer):
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    
    # Generate the response
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=200,
        num_return_sequences=1,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the question if it's repeated
    if response.lower().startswith(input_text.lower()):
        response = response[len(input_text):].strip()

    
    
    # Ensure the response ends with a proper sentence
    if not response.endswith(('.', '!', '?')):
        response = response.rsplit('.', 1)[0] + '.'
    
    return response """


    

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.json
    user_query = data.get('query')
    response = generate_response(user_query, model, tokenizer)
    return jsonify({'response': response})
    #return jsonify({ response})

if __name__ == '__main__':
    app.run(debug=True)





