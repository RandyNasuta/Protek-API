import os

import gradio as gr
import spaces
import uvicorn

from main import app as fastapi_app
from spaces.zero import startup as zerogpu_startup

@spaces.GPU(duration=10)
def check_status():
    return "API is running successfully"

with gr.Blocks() as demo:
    gr.Markdown("# Protek API Server")

    check_server = gr.Button("Server status check")
    result = gr.Textbox(label="Status")

    check_server.click(
        fn=check_status,
        inputs=[],
        outputs=result,
    )
    
app = gr.mount_gradio_app(
    fastapi_app,
    demo,
    path="/",
)
    
if __name__ == "__main__":
    print("Registering ZeroGPU startup...", flush=True)
    zerogpu_startup()

    print("Starting FastAPI and Gradio...", flush=True)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("APP_PORT") or 7860),
    )