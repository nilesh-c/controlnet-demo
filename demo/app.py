import io
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import Response
from PIL import Image
from model import BrainMRIImageGenerator

app = FastAPI()
generator = BrainMRIImageGenerator()

def process_and_generate_image(file: UploadFile, prompt: str, a_prompt: str, n_prompt: str):
    # Read image from the uploaded file
    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_image = np.array(image)

    # Generate image using the provided prompts
    generated_images = generator.generate_image(
        input_image=input_image,
        prompt=prompt,
        a_prompt=a_prompt,
        n_prompt=n_prompt
    )

    # Return the first generated image (edge map is the first element)
    output_image = Image.fromarray(generated_images[1])
    img_bytes = io.BytesIO()
    output_image.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    return img_bytes

@app.post("/generate")
async def generate_image(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    a_prompt: str = Form(...),
    n_prompt: str = Form(...)
):
    img_bytes = process_and_generate_image(
        file=file.file,
        prompt=prompt,
        a_prompt=a_prompt,
        n_prompt=n_prompt
    )
    return Response(content=img_bytes.read(), media_type="image/png")

@app.post("/generate_mri")
async def generate_mri_image():
    # Default prompts
    default_prompt = "mri brain scan"
    default_a_prompt = "good quality"
    default_n_prompt = "animal, drawing, painting, vivid colors, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"

    # Hardcoded path to the image
    image_path = "test_imgs/mri_brain.jpg"
    with open(image_path, "rb") as file:
        img_bytes = process_and_generate_image(
            file=file,
            prompt=default_prompt,
            a_prompt=default_a_prompt,
            n_prompt=default_n_prompt
        )
    return Response(content=img_bytes.read(), media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
