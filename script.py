import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoProcessor, AutoModelForCausalLM
from io import BytesIO
import uvicorn

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize model and processor (done once at startup)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Loading model on {device}...")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-base", 
    torch_dtype=torch_dtype, 
    trust_remote_code=True
).to(device)
processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-base", 
    trust_remote_code=True
)

@app.post("/generate-caption")
async def generate_caption(file: UploadFile = File(...)):
    try:
        # Read and process the image
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert('RGB')
        
        # Prepare inputs
        prompt = "<DETAILED_CAPTION>"
        inputs = processor(
            text=prompt, 
            images=image, 
            return_tensors="pt"
        ).to(device, torch_dtype)

        # Generate caption
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3,
        )
        
        generated_text = processor.batch_decode(
            generated_ids, 
            skip_special_tokens=False
        )[0]

        # Process the generated text
        parsed_answer = processor.post_process_generation(
            generated_text, 
            task="<DETAILED_CAPTION>", 
            image_size=(image.width, image.height)
        )

        # Return the caption
        return {
            "status": "success",
            "caption": parsed_answer['<DETAILED_CAPTION>']
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    # Run the server
    print("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)