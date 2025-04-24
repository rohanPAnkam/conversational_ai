import sys
import os
import uvicorn
import time
import logging
import multiprocessing

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from conversational_ai.controllers.api_controller import app
from conversational_ai.views.gradio_view import create_gradio_interface

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_fastapi():
    logger.info("Starting FastAPI server on port 8000...")
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        logger.error(f"FastAPI server failed to start: {str(e)}")
        raise

def run_gradio():
    logger.info("Starting Gradio interface...")
    try:
        demo = create_gradio_interface()
        if demo is None:
            logger.error("Gradio interface creation failed. Exiting...")
            return
        
        port = 7860
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                logger.info(f"Attempting to start Gradio on port {port}...")
                demo.launch(
                    server_name="0.0.0.0",
                    server_port=port,
                    share=False,
                    debug=True,
                    prevent_thread_lock=True
                )
                logger.info(f"Gradio interface started successfully on port {port}")
                break
            except Exception as e:
                logger.warning(f"Failed to start Gradio on port {port}: {str(e)}")
                port += 1
                if attempt == max_attempts - 1:
                    logger.error("Max attempts reached. Could not start Gradio interface.")
                    raise
    except Exception as e:
        logger.error(f"Gradio interface failed to start: {str(e)}")
        raise

if __name__ == "__main__":
    fastapi_process = multiprocessing.Process(target=run_fastapi)
    fastapi_process.start()
    
    logger.info("Waiting for FastAPI server to start...")
    time.sleep(5)
    
    run_gradio()
    
    try:
        fastapi_process.join()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        fastapi_process.terminate()
        fastapi_process.join()