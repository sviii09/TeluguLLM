from flask import Flask, request, jsonify
from flask_cors import CORS # To handle Cross-Origin Resource Sharing
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel # Only needed if you're loading fine-tuned PEFT adapters
import librosa
import numpy as np
import io
import base64
import tempfile
import os
import warnings

# Suppress some warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)
CORS(app) # Enable CORS for all routes, allowing frontend to access it

# --- Configuration ---
# Define the pre-trained model ID from Hugging Face Hub.
MODEL_ID = "Telugu-LLM-Labs/Indic-gemma-7b-finetuned-sft-Navarasa-2.0"
WHISPER_MODEL_ID = "openai/whisper-large-v3"  # Using the latest version for better performance

# --- Model Loading (This runs once when the Flask app starts) ---
print(f"Loading tokenizer for model: {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer loaded successfully.")

# Load Whisper model for speech-to-text
print(f"Loading Whisper model: {WHISPER_MODEL_ID}...")
try:
    whisper_processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_ID)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        WHISPER_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    whisper_model.eval()  # Set to evaluation mode
    print("Whisper model loaded successfully.")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    print("Falling back to whisper-large-v2...")
    try:
        WHISPER_MODEL_ID = "openai/whisper-large-v2"
        whisper_processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_ID)
        whisper_model = WhisperForConditionalGeneration.from_pretrained(
            WHISPER_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        whisper_model.eval()
        print("Whisper model (v2) loaded successfully.")
    except Exception as e2:
        print(f"Error loading Whisper v2: {e2}")
        print("Falling back to base whisper-large...")
        WHISPER_MODEL_ID = "openai/whisper-large"
        whisper_processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_ID)
        whisper_model = WhisperForConditionalGeneration.from_pretrained(
            WHISPER_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        whisper_model.eval()
        print("Whisper model (base) loaded successfully.")

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

# --- Speech-to-Text Function ---
def transcribe_audio(audio_data: bytes) -> str:
    """
    Transcribes audio data to text using Whisper model.
    Expects audio data as bytes.
    """
    try:
        # Create a temporary file to save the audio data
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            temp_audio.write(audio_data)
            temp_audio_path = temp_audio.name
        
        # Load audio using librosa
        try:
            audio_array, sampling_rate = librosa.load(temp_audio_path, sr=16000)
        except Exception as audio_load_error:
            print(f"Error loading audio with librosa: {audio_load_error}")
            # Clean up temporary file before re-raising
            os.unlink(temp_audio_path)
            raise audio_load_error
        
        # Clean up temporary file
        os.unlink(temp_audio_path)
        
        # Ensure audio is not empty
        if len(audio_array) == 0:
            raise ValueError("Audio file is empty or corrupted")
        
        # Normalize audio to prevent clipping
        audio_array = audio_array / np.max(np.abs(audio_array)) if np.max(np.abs(audio_array)) > 0 else audio_array
        
        # Process audio for Whisper
        input_features = whisper_processor(
            audio_array, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features.to(whisper_model.device)
        
        # Generate transcription
        # Force the model to transcribe in Telugu by setting the language
        try:
            forced_decoder_ids = whisper_processor.get_decoder_prompt_ids(
                language="te", 
                task="transcribe"
            )
        except Exception as decoder_error:
            print(f"Error getting decoder prompt IDs: {decoder_error}")
            # Fallback without forced language
            forced_decoder_ids = None
        
        # Generate with proper error handling
        generation_kwargs = {
            "input_features": input_features,
            "max_new_tokens": 448,
            "do_sample": False,  # Use greedy decoding for more consistent results
            "num_beams": 1,
        }
        
        if forced_decoder_ids is not None:
            generation_kwargs["forced_decoder_ids"] = forced_decoder_ids
        
        with torch.no_grad():  # Ensure no gradients are computed
            predicted_ids = whisper_model.generate(**generation_kwargs)
        
        # Decode the transcription
        transcription = whisper_processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )[0]
        
        # Clean up the transcription
        transcription = transcription.strip()
        
        # Remove common Whisper artifacts
        if transcription.lower().startswith("thank you"):
            transcription = ""
        
        print(f"Transcribed text: {transcription}")
        return transcription.strip()
        
    except Exception as e:
        print(f"Error during transcription: {e}")
        raise e

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

@app.route('/transcribe', methods=['POST'])
def transcribe_speech():
    """
    API endpoint to receive audio data and return transcribed text.
    Expects audio data in the request.
    """
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({"error": "No audio file selected"}), 400
        
        # Read audio data
        audio_data = audio_file.read()
        
        # Transcribe audio
        transcription = transcribe_audio(audio_data)
        
        return jsonify({"transcription": transcription})
        
    except Exception as e:
        print(f"Error during transcription: {e}")
        return jsonify({"error": "Internal server error during transcription"}), 500

@app.route('/speech-to-llm', methods=['POST'])
def speech_to_llm():
    """
    Combined endpoint: transcribe speech and generate LLM response.
    """
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({"error": "No audio file selected"}), 400
        
        # Read and transcribe audio
        audio_data = audio_file.read()
        transcription = transcribe_audio(audio_data)
        
        # Generate LLM response from transcription
        llm_response = generate_telugu_response_backend(transcription)
        
        return jsonify({
            "transcription": transcription,
            "response": llm_response
        })
        
    except Exception as e:
        print(f"Error during speech-to-LLM processing: {e}")
        return jsonify({"error": "Internal server error during speech processing"}), 500

# --- Run the Flask App ---
if __name__ == '__main__':
    # To run this, you might use: flask run
    # Or for development with automatic reload: flask --app backend_app run --debug
    # Or directly: python backend_app.py
    app.run(host='0.0.0.0', port=5000, debug=False) # Set debug=True for development
