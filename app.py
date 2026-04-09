from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
from pathlib import Path
import shutil
import uvicorn
import numpy as np
from PIL import Image
from pipelines.remotesam import RemoteSAMPipeline
from pipelines.sam3 import SAM3Pipeline
from pipelines.earthmind import EarthMindPipeline
from pipelines.image_classifier import classify_image_simple
import io
from api import generate_caption, answer_boolean_question, answer_numerical_question, answer_text_question, generate_bboxes, classify_prompt

app = FastAPI(title="ISRO MODELS PIPELINE BACKEND")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = None


@app.on_event("startup")
async def startup_event():
    global pipeline1, pipeline2, pipeline3
    pipeline1 = RemoteSAMPipeline(device="cuda")
    pipeline2 = SAM3Pipeline(device="cuda")
    pipeline3 = EarthMindPipeline(device="cuda")


@app.post("/classify-image/")
async def classify_image(
    image: UploadFile = File(...),
    device: str = Form("cuda")
):
    if not image.filename.endswith(('.png', '.jpg', '.jpeg', '.tiff')):
        raise HTTPException(status_code=400, detail="Invalid image format.")

    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / image.filename

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        classification_result = classify_image_simple(str(temp_path))
        return JSONResponse(content={"classification": classification_result})

    except Exception as e:
        print(f"Error classifying image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.post("/caption-query/")
async def caption_query(
    image: UploadFile = File(...),
    device: str = Form("cuda"),
    image_type: str = Form("optical")
):
    if not image.filename.endswith(('.png', '.jpg', '.jpeg', '.tiff')):
        raise HTTPException(status_code=400, detail="Invalid image format.")

    try:
        # Read file contents directly into memory
        contents = await image.read()

        # Open image from bytes and force load the image data
        temp_image = Image.open(io.BytesIO(contents))
        temp_image.load()  # Force PIL to load the full image data into memory

        caption_result = generate_caption(temp_image)
        return JSONResponse(content={"caption_result": caption_result})

    except Exception as e:
        print(f"Error generating caption: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/binary-query/")
async def binary_query(
    image: UploadFile = File(...),
    text_prompt: str = Form(...),
    device: str = Form("cuda"),
    image_type: str = Form("optical")
):
    if not image.filename.endswith(('.png', '.jpg', '.jpeg', '.tiff')):
        raise HTTPException(status_code=400, detail="Invalid image format.")

    try:
        # Read file contents directly into memory
        contents = await image.read()

        # Open image from bytes and force load the image data
        temp_image = Image.open(io.BytesIO(contents))
        temp_image.load()  # Force PIL to load the full image data into memory

        binary_result = answer_boolean_question(temp_image, text_prompt)
        return JSONResponse(content={"binary_result": binary_result})

    except Exception as e:
        print(f"Error in grounding: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/semantic-query/")
async def semantic_query(
    image: UploadFile = File(...),
    text_prompt: str = Form(...),
    device: str = Form("cuda"),
    image_type: str = Form("optical")
):
    if not image.filename.endswith(('.png', '.jpg', '.jpeg', '.tiff')):
        raise HTTPException(status_code=400, detail="Invalid image format.")

    try:
        # Read file contents directly into memory
        contents = await image.read()

        # Open image from bytes and force load the image data
        temp_image = Image.open(io.BytesIO(contents))
        temp_image.load()  # Force PIL to load the full image data into memory

        semantic_result = answer_text_question(temp_image, text_prompt)
        return JSONResponse(content={"semantic_result": semantic_result})

    except Exception as e:
        print(f"Error in semantic segmentation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/numeric-query/")
async def numeric_query(
    image: UploadFile = File(...),
    text_prompt: str = Form(...),
    device: str = Form("cuda"),
    image_type: str = Form("optical")
):
    if not image.filename.endswith(('.png', '.jpg', '.jpeg', '.tiff')):
        raise HTTPException(status_code=400, detail="Invalid image format.")

    try:
        # Read file contents directly into memory
        contents = await image.read()

        # Open image from bytes and force load the image data
        temp_image = Image.open(io.BytesIO(contents))
        temp_image.load()  # Force PIL to load the full image data into memory

        counting_result = answer_numerical_question(temp_image, text_prompt)
        return JSONResponse(content={"count": counting_result})

    except Exception as e:
        print(f"Error in counting: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/grounding-query/")
async def grounding_query(
    image: UploadFile = File(...),
    text_prompt: str = Form(...),
    device: str = Form("cuda"),
    image_type: str = Form("optical")
):
    if not image.filename.endswith(('.png', '.jpg', '.jpeg', '.tiff')):
        raise HTTPException(status_code=400, detail="Invalid image format.")

    try:
        # Read file contents directly into memory
        contents = await image.read()

        # Open image from bytes and force load the image data
        temp_image = Image.open(io.BytesIO(contents))
        temp_image.load()  # Force PIL to load the full image data into memory

        grounding_result = generate_bboxes(temp_image, text_prompt)

        return JSONResponse(content={"grounding": grounding_result})

    except Exception as e:
        print(f"Error in grounding query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/prompt-classifier/")
async def prompt_classifier(
    text_prompt: str = Form(...),
    device: str = Form("cuda")
):
    try:
        classification = pipeline1.classify_prompt(text_prompt=text_prompt)
        return JSONResponse(content={"identified_type": classification})

    except Exception as e:
        print(f"Error classifying prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/web-replier/")
async def web_replier(
    text_prompt: str = Form(...),
    image: UploadFile = File(None)
):
    try:
        temp_image = None
        image_type = "unknown"

        if image:
            if not image.filename.endswith(('.png', '.jpg', '.jpeg', '.tiff')):
                raise HTTPException(
                    status_code=400, detail="Invalid image format.")

            # Read file contents directly into memory
            contents = await image.read()
            # Open image from bytes and force load the image data
            temp_image = Image.open(io.BytesIO(contents))
            temp_image.load()  # Force PIL to load the full image data into memory

            # image_type = classify_image_simple(str(temp_path))

        prompt_type = classify_prompt(temp_image, text_prompt)
        # mapping = {
        #     'caption': 0,
        #     'grounding': 1,
        #     'binary': 2,
        #     'semantic': 3,
        #     'numeric': 4,
        # }
        if prompt_type == 0:
            result = generate_caption(temp_image)
        elif prompt_type == 1:
            result = generate_bboxes(temp_image, text_prompt)
        elif prompt_type == 2:
            result = answer_boolean_question(temp_image, text_prompt)
        elif prompt_type == 3:
            result = answer_text_question(temp_image, text_prompt)
        elif prompt_type == 4:
            result = answer_numerical_question(temp_image, text_prompt)
        else:
            result = generate_caption(temp_image)

        return JSONResponse(content={"response": result, "prompt_type": prompt_type})

    except Exception as e:
        print(f"Error generating web reply: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
