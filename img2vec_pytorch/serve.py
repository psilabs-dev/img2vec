from contextlib import asynccontextmanager
import io
import logging
import threading

from fastapi import FastAPI, UploadFile, File, status
from fastapi.responses import JSONResponse
from PIL import Image
from img2vec_pytorch import img_to_vec

LOGGER = logging.getLogger("uvicorn.img2vec")

class ModelWrapper:
    def __init__(self, model_instance):
        self.model = model_instance
        self.lock = threading.Lock()
wrapper = ModelWrapper(img_to_vec.Img2Vec())

@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        LOGGER.info("Initializing model.")
        wrapper.model.download_model()
    except Exception as e:
        LOGGER.error(f"Error initializing model: {e}")
    yield
    LOGGER.info("Application shutdown.")

app = FastAPI(lifespan=lifespan)

@app.get("/api/healthcheck")
async def get_healthcheck():
    return JSONResponse({
        "message": "OK"
    })

@app.post("/api/embeddings")
def get_embeddings(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        LOGGER.info("Successfully converted image from file.")
    except Exception as e:
        LOGGER.error(f"Error processing image: {e}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"message": "Invalid image file"}
        )
    try:
        with wrapper.lock:
            embeddings = wrapper.model.get_vec(img).tolist()
            LOGGER.info(f"Received embeddings: {embeddings}")
            return JSONResponse({
                "embeddings": embeddings
            })
    except Exception as e:
        LOGGER.exception(f"Error generating embeddings: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": "Internal Server Error"}
        )
