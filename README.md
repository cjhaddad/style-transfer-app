# Neural Style Transfer API Backend

A FastAPI-powered backend that applies artistic styles to uploaded images using deep learning neural networks.

## 🎨 Features

- **Real Neural Style Transfer** - Uses Google's pre-trained Magenta model via TensorFlow Hub
- **Multiple Artistic Styles** - Choose from famous paintings by Van Gogh, Monet, Picasso, and more
- **Fast Image Processing**
- **RESTful API**
- **Image Upload Support** - Handles JPEG, PNG formats with automatic preprocessing
- **Base64 Response** - Returns styled images ready for web display

## 🚀 Live Demo

Start the server and visit the interactive API documentation:
- **API Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 🏗️ Project Structure

```
Backend/
├── app.py                 # FastAPI application with all endpoints
├── style_transfer.py      # Neural style transfer logic
├── requirements.txt       # Python dependencies
├── static/
│   └── styles/           # Collection of artistic style images
└── venv/                 # Virtual environment
```

## 📋 Prerequisites

- **Python 3.8+** (tested with Python 3.8.9)
- **macOS 11.4+** with Apple Silicon (M1/M2) recommended
- **8GB+ RAM** for optimal TensorFlow performance

## 🛠️ Installation

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd style-transfer-app/backend
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Start Server
```bash
python app.py
```

Server starts at: http://localhost:8000

## 🔧 API Endpoints

### Health & Info
- `GET /` - Welcome message
- `GET /health` - Server health status

### Style Management  
- `GET /styles` - List all available artistic styles
- `GET /styles/{style_id}` - Download specific style image

### Image Processing
- `POST /upload-test` - Test image upload functionality
- `POST /style-transfer-basic` - Simple image blending (for testing)
- `POST /style-transfer` - **Neural style transfer** (main feature)

### Main Style Transfer Endpoint

**POST** `/style-transfer`

**Parameters:**
- `content_image`: Image file to stylize (multipart/form-data)
- `style_id`: ID of style to apply (form field)

**Response:**
```json
{
  "success": true,
  "message": "Neural style transfer completed using van_gogh_starry_night",
  "styled_image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "original_size": "1920x1080",
  "output_size": "512x384"
}
```

## 🎨 Included Style Images (1.5MB total)

This repository includes 7 curated artistic styles for immediate use:
- Van Gogh - Starry Night (388K)
- Seurat - A Sunday on La Grande Jatte (325K) 
- Hokusai - The Great Wave (284K)
- Kandinsky - Jaune Rouge Bleu (173K)
- Monet - Water Lilies (141K)
- Frida Kahlo - Self Portrait (104K)
- Georgia O'Keeffe - Series 1 (62K)

**Ready to use immediately** - no additional setup required!

## 🧠 Technical Details

### Neural Style Transfer
- **Model**: Google Magenta arbitrary-image-stylization-v1-256
- **Framework**: TensorFlow 2.9.0 with TensorFlow Hub
- **Optimization**: tensorflow-macos for Apple Silicon
- **Processing**: Content images resized to 512px, style images to 256px
- **Output**: High-quality JPEG with 90% quality

### Image Processing Pipeline
1. **Upload** → FastAPI receives multipart file
2. **Load** → PIL Image opens and validates format  
3. **Preprocess** → Resize, normalize, convert to tensors
4. **Transfer** → TensorFlow applies neural style transfer
5. **Postprocess** → Convert back to PIL Image
6. **Encode** → Base64 encoding for JSON response

## 🔍 Dependencies

### Core Framework
- **FastAPI 0.104.1** - Modern async web framework
- **Uvicorn 0.24.0** - ASGI server

### Image Processing  
- **Pillow 10.0.0** - Image manipulation library
- **python-multipart 0.0.6** - File upload support

### Machine Learning
- **tensorflow-macos 2.9.0** - Apple Silicon optimized TensorFlow
- **tensorflow-hub 0.12.0** - Pre-trained model access
- **numpy** - Numerical computing
