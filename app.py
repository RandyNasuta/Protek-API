import gradio as gr
import spaces

from main import app as fastapi_app

@spaces.GPU(duration=10)
def check_status():
    return "API is running successfully (ZeroGPU Bypassed)"

with gr.Blocks() as demo:
    gr.Markdown("# Protek API Server")
    
    check_server = gr.Button("Server status check")
    result = gr.Textbox(label="Status")
    
    check_server.click(
        fn=check_status,
        inputs=[],
        outputs=result
    )
    
app = gr.mount_gradio_app(
    fastapi_app,
    demo,
    path="/"
)