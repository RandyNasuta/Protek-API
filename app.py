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
    import subprocess

    print("Registering ZeroGPU startup...", flush=True)
    zerogpu_startup()

    print(f"APP_PORT = {os.environ.get('APP_PORT')}", flush=True)
    print(f"PORT = {os.environ.get('PORT')}", flush=True)

    print("Checking active processes:", flush=True)

    result = subprocess.run(
        ["sh", "-c", "ps -ef; (ss -ltnp || true)"],
        capture_output=True,
        text=True,
    )

    print(result.stdout, flush=True)
    print(result.stderr, flush=True)

    port = int(os.environ.get("APP_PORT") or 7860)

    print(f"Starting Uvicorn on port {port}...", flush=True)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
    )