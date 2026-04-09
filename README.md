# Language Guided SAM - Advanced AI/ML Pipeline for Remote Sensing Images Analysis

## 🌟 Key Features

- **Multi-Modal AI Models**: Integration of cutting-edge models including Qwen3-VL, Falcon, EarthMind, RemoteSAM, etc.
- **Advanced Pipelines**: Specialized pipelines for remote sensing tasks combining multiple AI models
- **Visual Question Answering**: Natural language querying capabilities for satellite imagery
- **Image Classification**: Automated classification between SAR and optical satellite imagery
- **FastAPI Backend**: Production-ready REST API for model serving
- **Comprehensive Evaluation**: Built-in metrics and testing frameworks for model performance assessment

## 🏗️ Architecture Overview

```
LG-SAM/
├── models/           # Model wrappers and implementations
├── pipelines/        # Task-specific pipeline combinations
├── vqa/              # Visual Question Answering modules
├── utils/            # Utility functions and metrics
├── app.py            # FastAPI application
├── api.py            # API endpoints
└── requirements.txt  # Dependencies
```

The framework follows a modular architecture where individual AI models are wrapped in standardized interfaces and combined into sophisticated pipelines for complex computer vision tasks.

## 🚀 Installation

### Automated Setup

```bash
# Run the setup script
chmod +x setup.sh
./setup.sh
```

The setup script will:

1. Create a virtual environment using `uv`
2. Install all dependencies
3. Clone and install SAM3
4. Extract model checkpoints
5. Start the FastAPI server

### API Usage

```bash
uv run uvicorn app:app --host 0.0.0.0 --port 8001
```

The API will be available at `http://localhost:8001`.

The API endpoints are:

- **POST `/classify-image/`** — Classifies the uploaded satellite image into SAR and OPTICAL.

- **POST `/caption-query/`** — Generates a descriptive caption for the uploaded image.

- **POST `/binary-query/`** — Answers yes/no (binary) questions about the uploaded image.

- **POST `/semantic-query/`** — Provides an answer to questions about scene features.

- **POST `/numeric-query/`** — Returns numerical answers such as counts or quantities from the image.

- **POST `/grounding-query/`** — Generates bounding boxes and grounding results based on the prompt and image content.

### Direct Pipeline Usage

```python
from pipelines.sgr import SGRPipeline

# Initialize pipeline
pipeline = SGRPipeline(device="cuda")

# Process image with text prompt
results = pipeline.process_image(
    image="satellite.jpg",
    text_prompt="locate the airport runway",
)

print(f"Found {len(results)} objects")
for result in results:
    print(f"Score: {result['score']:.3f}")
    print(f"Bounding box: {result['oriented_bbox']}")
```

## 📊 Evaluation

### Running Tests

```bash
# Test pipelines
python test_pipeline.py --annotations_dir data/vrsbench/annotations --images_dir data/vrsbench/images --num_gpus 4 --num_workers_per_gpu 2 --num_images 100 --batch_size 4 --viz_dir results

# VQA evaluation
python vqa/object_count.py annotations.json results.json data/vrsbench/images
```

---