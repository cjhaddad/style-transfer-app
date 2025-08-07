from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import os

# Create FastAPI app
app = FastAPI(
    title="Neural Style Transfer API",
    description="A simple API for neural style transfer",
    version="1.0.0"
)

@app.get("/")
def hello():
    return {"message": "Hello! Neural Style Transfer API is running!"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "Backend is working"}

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Endpoint for uploading images"""
        # Check if it's an image
    if not file.content_type.startswith('image/'):
        return {"error": "Please upload an image file"}
    content = await file.read()
    image = Image.open(io.BytesIO(content))

    return {
        "message": "Image uploaded successfully!",
        "filename": file.filename,
        "size": f"{image.width} x {image.height}",
        "format": image.format
    }

@app.get("/styles")
def get_available_styles():
    """Get list of available style images"""
    styles_dir = "static/styles"
    
    if not os.path.exists(styles_dir):
        return {"styles": [], "message": "No styles directory found"}
    
    styles = []
    for filename in os.listdir(styles_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Remove file extension for clean ID
            style_id = os.path.splitext(filename)[0]
            styles.append({
                "id": style_id,
                "name": style_id.replace("_", " ").title(),
                "filename": filename
            })
    
    return {"styles": styles, "count": len(styles)}

# This will only run if you execute this file directly
if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)