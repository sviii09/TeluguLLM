# TeluguLLM

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

### Option 1: Gradio Interface (Original)
```bash
python LLM.py
```

### Option 2: Web Interface with Speech-to-Text
1. Start the Flask backend:
```bash
python backend.py
```

2. Open `demo.html` in your web browser

## Features

- **Text Input**: Type your questions in Telugu
- **Speech-to-Text**: Click the microphone button to speak your question in Telugu
- **Telugu LLM**: Get responses from a fine-tuned Telugu language model
- **Web Interface**: Modern, responsive web interface

## Speech-to-Text Integration

The application now includes OpenAI's Whisper large model for speech-to-text functionality:
- Supports Telugu speech recognition
- Real-time audio recording
- Automatic transcription and LLM response generation
- Works directly in the browser (requires microphone permissions)

## API Endpoints

- `POST /generate` - Generate text response from Telugu input
- `POST /transcribe` - Transcribe audio to Telugu text
- `POST /speech-to-llm` - Combined speech-to-text and LLM response

## Browser Requirements

For speech functionality, you need:
- Modern browser with microphone support
- HTTPS connection (for production) or localhost (for development)
- Microphone permissions granted
