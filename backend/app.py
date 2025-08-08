from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from PIL import Image
import io
import os
import base64
from neural_style_transfer import neural_style_transfer

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

@app.get("/styles/{style_id}")
def get_style_image(style_id: str):
    """Serve a specific style image"""
    styles_dir = "static/styles"

    # Try different file extensions
    for ext in ['.jpg', '.jpeg', '.png']:
        file_path = os.path.join(styles_dir, f"{style_id}{ext}")
        if os.path.exists(file_path):
            return FileResponse(file_path)

    return {"error": f"Style image '{style_id}' not found"}

@app.post("/style-transfer")
async def neural_style_transfer_endpoint(
    content_image: UploadFile = File(...),
    style_id: str = Form(...)
):
    """Real neural style transfer using TensorFlow Hub"""

    # Validate inputs
    if not content_image.content_type.startswith('image/'):
        return {"error": "Please upload an image file"}

    # Find style image
    styles_dir = "static/styles"
    style_path = None
    for ext in ['.jpg', '.jpeg', '.png']:
        potential_path = os.path.join(styles_dir, f"{style_id}{ext}")
        if os.path.exists(potential_path):
            style_path = potential_path
            break

    if not style_path:
        return {"error": f"Style '{style_id}' not found"}

    try:
        print(f"🎨 Processing neural style transfer with style: {style_id}")

        # Load content image
        content_data = await content_image.read()
        content_img = Image.open(io.BytesIO(content_data)).convert('RGB')
        print(f"📸 Content image loaded: {content_img.size}")

        # Load style image
        style_img = Image.open(style_path).convert('RGB')
        print(f"🖼️  Style image loaded: {style_img.size}")

        # Perform neural style transfer
        stylized_image = neural_style_transfer.transfer_style(
            content_image=content_img,
            style_image=style_img,
            content_size=512,  # Adjust for speed vs quality
            style_size=256
        )

        # Convert to base64 for response
        img_buffer = io.BytesIO()
        stylized_image.save(img_buffer, format='JPEG', quality=90)
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()

        return {
            "success": True,
            "message": f"Neural style transfer completed using {style_id}",
            "styled_image": f"data:image/jpeg;base64,{img_base64}",
            "original_size": f"{content_img.width}x{content_img.height}",
            "output_size": f"{stylized_image.width}x{stylized_image.height}"
        }

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {"error": f"Neural style transfer failed: {str(e)}"}
# This will only run if you execute this file directly
if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)