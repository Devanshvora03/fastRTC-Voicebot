import base64
import json
import os
from pathlib import Path
from io import BytesIO
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastrtc import (
    AdditionalOutputs,
    ReplyOnPause,
    Stream,
    get_cloudflare_turn_credentials_async,
    get_stt_model,
)
from gradio.utils import get_space
from pydantic import BaseModel
from groq import AsyncGroq
import gradio as gr
from scipy.io.wavfile import read, write

# Load environment variables
load_dotenv()

curr_dir = Path(__file__).parent

# Check for Groq API key
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

# Initialize Groq client and STT model
client = AsyncGroq(api_key=groq_api_key)
stt_model = get_stt_model()

async def text_to_speech(text: str) -> tuple[int, np.ndarray]:
    """Convert text to speech using Groq's playai-tts model, with gTTS fallback."""
    try:
        response = await client.audio.speech.create(
            model="playai-tts",
            voice="Celeste-PlayAI",  # Natural English voice
            input=text,
            response_format="wav"
        )
        # Read audio data
        audio_data = await response.read()

        # Convert WAV to numpy array
        from pydub import AudioSegment
        audio = AudioSegment.from_wav(BytesIO(audio_data))
        audio = audio.set_frame_rate(22050).set_channels(1)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
        if not np.isfinite(samples).all():
            raise ValueError("Invalid audio samples from playai-tts")
        print(f"playai-tts success: sample_rate=22050, samples_shape={samples.shape}")
        return 22050, samples
    except Exception as e:
        if "model_terms_required" in str(e):
            print("Error: playai-tts requires terms acceptance at https://console.groq.com/playground?model=playai-tts. Falling back to gTTS.")
        else:
            print(f"Groq TTS error: {e}. Falling back to gTTS.")

        # Fallback to gTTS
        try:
            from gtts import gTTS
            from pydub import AudioSegment
            mp3_buffer = BytesIO()
            tts = gTTS(text=text, lang="en")
            tts.write_to_fp(mp3_buffer)
            mp3_buffer.seek(0)
            audio = AudioSegment.from_mp3(mp3_buffer)
            audio = audio.set_frame_rate(22050).set_channels(1)
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
            if not np.isfinite(samples).all():
                raise ValueError("Invalid audio samples from gTTS")
            print(f"gTTS success: sample_rate=22050, samples_shape={samples.shape}")
            return 22050, samples
        except Exception as gtts_e:
            print(f"gTTS error: {gtts_e}. Ensure FFmpeg is installed.")
            sample_rate = 22050
            samples = np.zeros(sample_rate, dtype=np.float32)
            print("Both TTS methods failed. Returning empty audio.")
            return sample_rate, samples

async def response(
    audio: tuple[int, np.ndarray],
    gradio_chatbot: list[dict] | None = None,
    conversation_state: list[dict] | None = None,
):
    gradio_chatbot = gradio_chatbot or []
    conversation_state = conversation_state or []

    # Convert input audio to text
    text = stt_model.stt(audio)
    sample_rate, array = audio
    gradio_chatbot.append(
        {"role": "user", "content": gr.Audio((sample_rate, array.squeeze()))}
    )
    yield AdditionalOutputs(gradio_chatbot, conversation_state)

    conversation_state.append({"role": "user", "content": text})
    try:
        # Get LLM response
        request = await client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=conversation_state,
            temperature=0.1,
            top_p=0.1,
        )
        response_text = request.choices[0].message.content
        response = {"role": "assistant", "content": response_text}
    except Exception as e:
        print(f"Groq LLM error: {e}")
        response_text = "Sorry, I ran into an issue."
        response = {"role": "assistant", "content": response_text}

    conversation_state.append(response)

    # Convert assistant response to speech
    tts_sample_rate, tts_audio = await text_to_speech(response_text)
    if tts_audio.size > 0 and np.isfinite(tts_audio).all():
        gradio_chatbot.append(
            {"role": "assistant", "content": gr.Audio((tts_sample_rate, tts_audio))}
        )
    else:
        gradio_chatbot.append({"role": "assistant", "content": response_text})

    yield AdditionalOutputs(gradio_chatbot, conversation_state)

chatbot = gr.Chatbot(type="messages", value=[])
state = gr.State(value=[])
stream = Stream(
    ReplyOnPause(
        response,
        input_sample_rate=16000,
    ),
    mode="send",
    modality="audio",
    additional_inputs=[chatbot, state],
    additional_outputs=[chatbot, state],
    additional_outputs_handler=lambda *a: (a[2], a[3]),
    concurrency_limit=20 if get_space() else None,
    rtc_configuration=get_cloudflare_turn_credentials_async,
)

app = FastAPI()
stream.mount(app)

class Message(BaseModel):
    role: str
    content: str

class InputData(BaseModel):
    webrtc_id: str
    chatbot: list[Message]
    state: list[Message]

@app.get("/")
async def _():
    try:
        rtc_config = await get_cloudflare_turn_credentials_async(
            hf_token=os.getenv("HF_TOKEN_ALT")
        )
        html_content = (curr_dir / "index.html").read_text(encoding="utf-8")
        html_content = html_content.replace("__RTC_CONFIGURATION__", json.dumps(rtc_config))
        return HTMLResponse(content=html_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading HTML: {str(e)}")

@app.post("/input_hook")
async def _(data: InputData):
    body = data.model_dump()
    stream.set_input(data.webrtc_id, body["chatbot"], body["state"])

def audio_to_base64(audio_data):
    """Convert audio data to base64 WAV with robust format handling."""
    try:
        # Handle Gradio Audio component or FileData dict
        if hasattr(audio_data, 'value'):
            audio_data = audio_data.value
        if isinstance(audio_data, dict) and 'path' in audio_data:
            return audio_to_base64(load_audio_file(audio_data['path']))

        # Handle tuple/numpy array input
        if isinstance(audio_data, tuple):
            if len(audio_data) == 2:
                sample_rate, samples = audio_data
            else:
                sample_rate = audio_data[0] if len(audio_data) > 0 else 22050
                samples = audio_data[1] if len(audio_data) > 1 else np.zeros(0)
        else:
            sample_rate = 22050
            samples = audio_data if isinstance(audio_data, np.ndarray) else np.array(audio_data)

        # Ensure proper array format
        samples = np.array(samples, dtype=np.float32).squeeze()
        if samples.ndim > 1:
            samples = samples.mean(axis=1)

        # Validate and convert to WAV
        if len(samples) == 0 or not np.isfinite(samples).all():
            print("Invalid audio data. Returning None.")
            return None

        samples_int16 = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
        buffer = BytesIO()
        write(buffer, sample_rate, samples_int16)
        return f"data:audio/wav;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"

    except Exception as e:
        print(f"audio_to_base64 error: {str(e)}")
        return None

def load_audio_file(file_path: str) -> tuple[int, np.ndarray]:
    """Load audio file with robust error handling."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        sample_rate, samples = read(file_path)
        samples = samples.astype(np.float32) / 32768.0

        if samples.ndim > 1:
            samples = samples.mean(axis=1)

        if not np.isfinite(samples).all():
            raise ValueError("Invalid audio samples in file")

        return sample_rate, samples
    except Exception as e:
        print(f"load_audio_file error: {str(e)}")
        return 22050, np.zeros(0, dtype=np.float32)

@app.get("/outputs")
async def _(webrtc_id: str):
    async def output_stream():
        try:
            async for output in stream.output_stream(webrtc_id):
                try:
                    chatbot = output.args[0]
                    state = output.args[1]

                    if not chatbot or len(chatbot) == 0:
                        print("Empty chatbot list")
                        continue

                    last_chat = chatbot[-1]
                    print(f"Processing message: role={last_chat['role']}, content_type={type(last_chat['content'])}")

                    audio = None

                    if isinstance(last_chat["content"], gr.Audio):
                        print(f"Content is gr.Audio, value_type={type(last_chat['content'].value)}")
                        try:
                            if last_chat["role"] == "user":
                                file_data = last_chat["content"].value
                                print(f"User audio data: {type(file_data)}")
                                audio = audio_to_base64(file_data)
                            else:
                                value = last_chat["content"].value
                                print(f"Assistant audio data: {type(value)}")
                                audio = audio_to_base64(value)
                        except Exception as e:
                            import traceback
                            print(f"Error processing audio in outputs: {e}")
                            traceback.print_exc()
                            audio = None

                    data = {
                        "message": state[-1] if state and len(state) > 0 else {"role": "assistant", "content": "No message available"},
                        "audio": audio
                    }

                    yield f"event: output\ndata: {json.dumps(data)}\n\n"
                except Exception as e:
                    import traceback
                    print(f"Error in output processing: {e}")
                    traceback.print_exc()
                    yield f"event: output\ndata: {json.dumps({'message': {'role': 'assistant', 'content': 'Error processing response'}, 'audio': None})}\n\n"
        except Exception as e:
            import traceback
            print(f"Error in output stream: {e}")
            traceback.print_exc()

    return StreamingResponse(output_stream(), media_type="text/event-stream")

@app.get("/api-status")
async def check_api_status():
    try:
        test_response = await client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        return {"status": "ok", "message": "API key is valid"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/tts-test")
async def tts_test():
    sample_rate, audio = await text_to_speech("Hello, this is a test.")
    return {"status": "ok" if audio.size > 0 and np.isfinite(audio).all() else "error"}

if __name__ == "__main__":
    if groq_api_key:
        print(f"GROQ_API_KEY found with length: {len(groq_api_key)}")
    else:
        print("WARNING: GROQ_API_KEY not found")

    if (mode := os.getenv("MODE")) == "UI":
        stream.ui.launch(server_port=7860, share=True)
    elif mode == "PHONE":
        raise ValueError("Phone mode not supported")
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=7860)