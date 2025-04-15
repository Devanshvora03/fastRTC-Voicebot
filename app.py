import base64
import json
import os
from pathlib import Path

import gradio as gr
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

# Load environment variables from .env file
load_dotenv()

curr_dir = Path(__file__).parent

# Check for Groq API key and initialize client
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set. Please add it to your .env file.")

# Initialize Groq client with explicit API key
client = AsyncGroq(api_key=groq_api_key)
stt_model = get_stt_model()


async def response(
    audio: tuple[int, np.ndarray],
    gradio_chatbot: list[dict] | None = None,
    conversation_state: list[dict] | None = None,
):
    gradio_chatbot = gradio_chatbot or []
    conversation_state = conversation_state or []
    print("chatbot", gradio_chatbot)

    text = stt_model.stt(audio)
    sample_rate, array = audio
    gradio_chatbot.append(
        {"role": "user", "content": gr.Audio((sample_rate, array.squeeze()))}
    )
    yield AdditionalOutputs(gradio_chatbot, conversation_state)

    conversation_state.append({"role": "user", "content": text})
    try:
        # Use Groq client with proper error handling
        request = await client.chat.completions.create(
            model="llama3-8b-8192",  # Groq compatible model
            messages=conversation_state,
            temperature=0.1,
            top_p=0.1,
        )
        response = {"role": "assistant", "content": request.choices[0].message.content}
    except Exception as e:
        print(f"Error from Groq API: {e}")
        response = {"role": "assistant", "content": f"I'm sorry, I encountered an error processing your request. Please check your API key and try again."}

    conversation_state.append(response)
    gradio_chatbot.append(response)

    yield AdditionalOutputs(gradio_chatbot, conversation_state)


chatbot = gr.Chatbot(type="messages", value=[])
state = gr.State(value=[])
stream = Stream(
    ReplyOnPause(
        response,  # type: ignore
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
    rtc_config = await get_cloudflare_turn_credentials_async(
        hf_token=os.getenv("HF_TOKEN_ALT")
    )
    html_content = (curr_dir / "index.html").read_text(encoding="utf-8")  # Specify UTF-8 encoding
    html_content = html_content.replace("__RTC_CONFIGURATION__", json.dumps(rtc_config))
    return HTMLResponse(content=html_content)


@app.post("/input_hook")
async def _(data: InputData):
    body = data.model_dump()
    stream.set_input(data.webrtc_id, body["chatbot"], body["state"])


def audio_to_base64(file_path):
    audio_format = "wav"
    with open(file_path, "rb") as audio_file:
        encoded_audio = base64.b64encode(audio_file.read()).decode("utf-8")
    return f"data:audio/{audio_format};base64,{encoded_audio}"


@app.get("/outputs")
async def _(webrtc_id: str):
    async def output_stream():
        async for output in stream.output_stream(webrtc_id):
            chatbot = output.args[0]
            state = output.args[1]
            data = {
                "message": state[-1],
                "audio": audio_to_base64(chatbot[-1]["content"].value["path"])
                if chatbot[-1]["role"] == "user"
                else None,
            }
            yield f"event: output\ndata: {json.dumps(data)}\n\n"

    return StreamingResponse(output_stream(), media_type="text/event-stream")


@app.get("/api-status")
async def check_api_status():
    """Endpoint to verify API key is working"""
    try:
        # Simple test call to check if API key is valid
        test_response = await client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        return {"status": "ok", "message": "API key is valid"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import os

    # Print confirmation that we have loaded the API key (without showing the key itself)
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