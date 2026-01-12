# Clothing Recolor API

A FastAPI server that uses YOLO11 segmentation to automatically recolor clothing in images while preserving texture and shading.

## Features

- 🎨 **Automatic clothing segmentation** using YOLO11
- 🖼️ **Texture-preserving recoloring** via HSV color space manipulation
- 🚀 **Fast inference** optimized for CPU (VPS-friendly)
- 🌐 **REST API** with CORS support for web integration
- ⚛️ **React/TypeScript components** included

## Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd RecolorOutfitInference

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the Server

```bash
python server.py
```

Server starts at `http://localhost:8000`

### API Documentation

Interactive docs available at `http://localhost:8000/docs`

## API Endpoints

### POST `/recolor`

Recolor clothing in an image.

**Request:**
- `image`: Image file (JPG, PNG)
- `color`: Hex color code (`#RRGGBB`)

**Response:** JPG image

**Example (cURL):**
```bash
curl -X POST "http://localhost:8000/recolor" \
  -F "image=@person.jpg" \
  -F "color=#0066CC" \
  --output recolored.jpg
```

**Example (Python):**
```python
import requests

with open('person.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/recolor',
        files={'image': f},
        data={'color': '#FF0000'}
    )

with open('result.jpg', 'wb') as f:
    f.write(response.content)
```

### GET `/health`

Health check endpoint.

**Response:**
```json
{"status": "ok"}
```

## React Integration

### Install the API client

Copy the files from `frontend/` to your React project:

```
frontend/
├── recolorApi.ts          # API client
├── ProductColorPicker.tsx # Ready-to-use component
└── ExampleUsage.tsx       # Integration examples
```

### Usage

```tsx
import { ProductColorPicker } from './ProductColorPicker';

function ProductPage({ product }) {
  return (
    <ProductColorPicker
      originalImageUrl={product.imageUrl}
      onImageChange={(newUrl) => console.log('New image:', newUrl)}
    />
  );
}
```

### Environment Variable

Set the API URL in your React app:

```env
REACT_APP_RECOLOR_API_URL=http://your-server:8000
```

## How It Works

1. **Person Segmentation**: YOLO11 detects and segments the person in the image
2. **Skin Detection**: HSV-based skin detection excludes face/hands from the mask
3. **Clothing Mask**: `clothing = person - skin`
4. **HSV Recoloring**: Changes hue/saturation while preserving original brightness (texture)
5. **Edge Blending**: Gaussian blur on mask edges for seamless blending

## Project Structure

```
RecolorOutfitInference/
├── server.py              # FastAPI server
├── main.py                # CLI version (standalone)
├── requirements.txt       # Python dependencies
├── frontend/              # React/TypeScript integration
│   ├── recolorApi.ts
│   ├── ProductColorPicker.tsx
│   └── ExampleUsage.tsx
└── README.md
```

## Requirements

- Python 3.10+
- ~2GB RAM (for YOLO11-medium model)
- CPU only (no GPU required)

## Dependencies

- FastAPI + Uvicorn
- OpenCV
- Ultralytics (YOLO11)
- NumPy

## Configuration

The server uses `yolo11m-seg.pt` (medium) by default. For faster inference on limited hardware, edit `server.py`:

```python
# Change from medium to nano model
YOLO('yolo11n-seg.pt')  # Faster, less accurate
```

## License

MIT
