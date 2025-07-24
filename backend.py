from flask import Flask, request, jsonify
from flask_cors import CORS # To handle Cross-Origin Resource Sharing
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel # Only needed if you're loading fine-tuned PEFT adapters

app = Flask(__name__)
CORS(app) # Enable CORS for all routes, allowing frontend to access it

# --- Configuration ---
# Define the pre-trained model ID from Hugging Face Hub.
MODEL_ID = "Telugu-LLM-Labs/Indic-gemma-7b-finetuned-sft-Navarasa-2.0"

# --- Model Loading (This runs once when the Flask app starts) ---
print(f"Loading tokenizer for model: {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer loaded successfully.")

print("Configuring BitsAndBytes for quantization...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 # Use bfloat16 if your GPU supports it
)
print("BitsAndBytes configuration complete.")

print(f"Loading model: {MODEL_ID} with quantization...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model with quantization: {e}")
    print("Attempting to load model without quantization (might require more VRAM)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16 # Fallback
    )
    print("Model loaded successfully without quantization.")

model.eval() # Set model to evaluation mode

# --- Inference Function (Backend Logic) ---
def generate_telugu_response_backend(user_input: str) -> str:
    """
    Generates a Telugu response from the LLM based on user input.
    This function is called by the API endpoint.
    """
    prompt = f"### Instruction:\n{user_input}\n\n### Response:"
    print(f"Received input for LLM: {user_input}")

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=250,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    response_start_marker = "### Response:"
    if response_start_marker in generated_text:
        response_only = generated_text.split(response_start_marker, 1)[1].strip()
    else:
        response_only = generated_text.strip()

    response_only = response_only.replace("### Instruction:", "").replace("### Response:", "").strip()
    print(f"Generated response from LLM: {response_only}")
    return response_only

# --- API Endpoint ---
@app.route('/generate', methods=['POST'])
def generate_text():
    """
    API endpoint to receive user input and return LLM generated text.
    Expects a JSON payload with a 'text' key.
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    user_input = data.get('text')

    if not user_input:
        return jsonify({"error": "Missing 'text' in request body"}), 400

    try:
        llm_response = generate_telugu_response_backend(user_input)
        return jsonify({"response": llm_response})
    except Exception as e:
        print(f"Error during LLM generation: {e}")
        return jsonify({"error": "Internal server error during text generation"}), 500

# --- Run the Flask App ---
if __name__ == '__main__':
    # To run this, you might use: flask run
    # Or for development with automatic reload: flask --app backend_app run --debug
    # Or directly: python backend_app.py
    app.run(host='0.0.0.0', port=5000, debug=False) # Set debug=True for development
