from contextlib import asynccontextmanager
import io
import logging
import os
import threading
from typing import Annotated, List, Optional, TypeAlias

import dotenv
from fastapi import Depends, FastAPI, UploadFile, status
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from img2vec_pytorch import img_to_vec

LOGGER = logging.getLogger("uvicorn.img2vec")
class SuppressHealthcheckFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool | logging.LogRecord:
        return record.getMessage().find("/api/healthcheck") == -1
logging.getLogger("uvicorn.access").addFilter(SuppressHealthcheckFilter())

class ApplicationConfiguration:
    def __init__(self):
        dotenv.load_dotenv()
        self.model_name = os.getenv("IMG2VEC_MODEL", 'resnet-18')
        model_layer = os.getenv("IMG2VEC_MODEL_LAYER", 'default')
        if model_layer.isdigit():
            model_layer = int(model_layer)
        self.model_layer = model_layer
        layer_output_size = os.getenv("IMG2VEC_LAYER_OUTPUT_SIZE", '512')
        if not layer_output_size.isdigit():
            raise TypeError(f"IMG2VEC_LAYER_OUTPUT_SIZE {layer_output_size} is invalid.")
        self.layer_output_size = int(layer_output_size)
        gpu = os.getenv("IMG2VEC_GPU", "0")
        self.gpu = 0 if not gpu.isdigit() else int(gpu)
        device_preference = os.getenv("IMG2VEC_DEVICES", "cuda, mps, cpu")
        self.device_preference = [d.strip() for d in device_preference.split(',')]

config = ApplicationConfiguration()
def get_config():
    return config

class CreateEmbeddingsResponse(BaseModel):
    error: Optional[str] = None
    embeddings: Optional[List[float]] = None

class ModelContext:
    def __init__(self, img2vec: img_to_vec.AbstractImg2Vec):
        self.img2vec = img2vec
        self.lock = threading.Lock()

model_context = ModelContext(img_to_vec.Img2Vec(
    model=config.model_name,
    layer=config.model_layer,
    layer_output_size=config.layer_output_size,
    gpu=config.gpu,
    device_preference=config.device_preference
))

def get_model_context():
    return model_context
ModelContextT: TypeAlias = Annotated[ModelContext, Depends(get_model_context)]

@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        LOGGER.info("Initializing model.")
        model_context.img2vec.download_model()
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
def get_embeddings(model_context: ModelContextT, file: UploadFile):
    try:
        contents = file.file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        LOGGER.debug("Successfully converted image from file.")
    except Exception as e:
        LOGGER.error(f"Error processing image: {e}")
        response = CreateEmbeddingsResponse(error="Invalid image file.")
        return JSONResponse(response.model_dump(), status_code=status.HTTP_400_BAD_REQUEST)
    try:
        with model_context.lock:
            embeddings = model_context.img2vec.get_vec(img).tolist()
            LOGGER.debug(f"Got embeddings: {embeddings}")
            response = CreateEmbeddingsResponse(embeddings=embeddings)
            return JSONResponse(response.model_dump())
    except Exception as e:
        LOGGER.exception(f"Error generating embeddings: {e}")
        response = CreateEmbeddingsResponse(error="Internal Server Error")
        return JSONResponse(response.model_dump(), status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
