"""
High-Performance Inference Server for RF-DETR Surveillance
FastAPI-based serving infrastructure with real-time processing
"""
import logging
import asyncio
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, AsyncGenerator
from dataclasses import dataclass, field
import json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import io
import base64
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque

try:
    from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Depends
    from fastapi.responses import JSONResponse, StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logging.warning("FastAPI not available. Server functionality disabled.")

try:
    import onnxruntime as ort
    ONNX_RUNTIME_AVAILABLE = True
except ImportError:
    ONNX_RUNTIME_AVAILABLE = False
    logging.debug("ONNX Runtime not available.")

from ..models.surveillance_detector import create_surveillance_detector, CrowdAwareDetector
from .model_export import ModelExporter

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Configuration for inference server."""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    # Model settings
    model_path: Path = field(default_factory=lambda: Path("model.pt"))
    model_type: str = "torchscript"  # "torchscript", "onnx", "tensorrt"
    device: str = "auto"  # "auto", "cpu", "cuda"
    batch_size: int = 1
    
    # Processing settings
    max_image_size: int = 1024
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    max_detections: int = 100
    
    # Performance settings
    enable_batching: bool = True
    batch_timeout_ms: int = 10
    max_batch_size: int = 8
    async_processing: bool = True
    
    # Monitoring settings
    enable_metrics: bool = True
    metrics_retention_hours: int = 24
    log_level: str = "INFO"
    
    # Security settings
    enable_cors: bool = True
    max_file_size_mb: int = 50
    rate_limit_requests_per_minute: int = 1000


# Pydantic models for API
class DetectionRequest(BaseModel):
    """Request model for detection API."""
    
    image_data: Optional[str] = Field(None, description="Base64 encoded image")
    image_url: Optional[str] = Field(None, description="URL to image")
    confidence_threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0)
    nms_threshold: Optional[float] = Field(0.4, ge=0.0, le=1.0)
    max_detections: Optional[int] = Field(100, ge=1, le=1000)
    return_features: bool = Field(False, description="Return feature maps")
    request_id: Optional[str] = Field(None, description="Request identifier")


class BoundingBox(BaseModel):
    """Bounding box model."""
    
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str


class DetectionResult(BaseModel):
    """Detection result model."""
    
    request_id: str
    detections: List[BoundingBox]
    processing_time_ms: float
    image_shape: Tuple[int, int]  # (height, width)
    model_info: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ServerStatus(BaseModel):
    """Server status model."""
    
    status: str
    model_loaded: bool
    device: str
    model_type: str
    uptime_seconds: float
    requests_processed: int
    average_processing_time_ms: float
    current_load: float
    memory_usage_mb: float


class PerformanceMetrics:
    """Performance metrics collection."""
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.request_times = deque(maxlen=10000)
        self.processing_times = deque(maxlen=10000)
        self.error_counts = defaultdict(int)
        self.start_time = time.time()
        
    def add_request(self, processing_time: float, success: bool = True):
        """Add request metrics."""
        current_time = time.time()
        self.request_times.append(current_time)
        
        if success:
            self.processing_times.append(processing_time)
        else:
            self.error_counts['failed_requests'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        current_time = time.time()
        
        # Filter recent data
        recent_cutoff = current_time - (self.retention_hours * 3600)
        recent_times = [t for t in self.request_times if t > recent_cutoff]
        recent_processing = [t for t in self.processing_times if t is not None]
        
        stats = {
            'uptime_seconds': current_time - self.start_time,
            'total_requests': len(self.request_times),
            'recent_requests': len(recent_times),
            'requests_per_hour': len(recent_times) / max(self.retention_hours, 1),
            'average_processing_time_ms': np.mean(recent_processing) * 1000 if recent_processing else 0,
            'p95_processing_time_ms': np.percentile(recent_processing, 95) * 1000 if len(recent_processing) > 10 else 0,
            'error_rate': sum(self.error_counts.values()) / max(len(self.request_times), 1),
            'current_load': len(recent_times) / max(len(self.request_times), 1)
        }
        
        return stats


class ModelManager:
    """Model loading and inference management."""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.model = None
        self.detector = None
        self.device = self._get_device()
        self.model_info = {}
        
    def _get_device(self) -> torch.device:
        """Determine inference device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        else:
            return torch.device(self.config.device)
    
    def load_model(self) -> bool:
        """Load model for inference."""
        try:
            model_path = self.config.model_path
            
            if not model_path.exists():
                logger.error(f"Model not found: {model_path}")
                return False
            
            logger.info(f"Loading model from {model_path}")
            logger.info(f"Model type: {self.config.model_type}")
            logger.info(f"Device: {self.device}")
            
            if self.config.model_type == "torchscript":
                self.model = torch.jit.load(model_path, map_location=self.device)
                self.model.eval()
                
            elif self.config.model_type == "onnx" and ONNX_RUNTIME_AVAILABLE:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device.type == 'cuda' else ['CPUExecutionProvider']
                self.model = ort.InferenceSession(str(model_path), providers=providers)
                
            else:
                raise ValueError(f"Unsupported model type: {self.config.model_type}")
            
            # Initialize detector
            self.detector = create_surveillance_detector(
                confidence_threshold=self.config.confidence_threshold,
                person_confidence_threshold=self.config.confidence_threshold * 0.8,
                scale_aware_nms=True
            )
            
            # Store model info
            self.model_info = {
                'model_path': str(model_path),
                'model_type': self.config.model_type,
                'device': str(self.device),
                'file_size_mb': model_path.stat().st_size / (1024 * 1024)
            }
            
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def predict(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Run inference on image."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Run inference
            if self.config.model_type == "torchscript":
                with torch.no_grad():
                    predictions = self.model(processed_image)
            
            elif self.config.model_type == "onnx":
                input_name = self.model.get_inputs()[0].name
                ort_inputs = {input_name: processed_image.cpu().numpy()}
                ort_outputs = self.model.run(None, ort_inputs)
                
                # Convert back to dict format
                predictions = {
                    'pred_logits': torch.from_numpy(ort_outputs[0]),
                    'pred_boxes': torch.from_numpy(ort_outputs[1]) if len(ort_outputs) > 1 else torch.empty(0)
                }
            
            else:
                raise ValueError(f"Unsupported model type for inference: {self.config.model_type}")
            
            # Post-process predictions
            image_sizes = [(image.shape[0], image.shape[1])]  # (H, W)
            detection_results = self.detector(predictions, image_sizes)
            
            processing_time = time.time() - start_time
            
            return {
                'detections': detection_results[0],
                'processing_time': processing_time,
                'image_shape': image.shape[:2],
                'model_info': self.model_info
            }
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR, convert to RGB
            image = image[:, :, ::-1]
        
        # Resize if too large
        h, w = image.shape[:2]
        if max(h, w) > self.config.max_image_size:
            scale = self.config.max_image_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image = np.array(Image.fromarray(image).resize((new_w, new_h)))
        
        # Normalize to [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # NCHW
        
        # Move to device
        image_tensor = image_tensor.to(self.device)
        
        return image_tensor


class InferenceServer:
    """High-performance inference server for RF-DETR surveillance."""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.model_manager = ModelManager(config)
        self.metrics = PerformanceMetrics(config.metrics_retention_hours)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize FastAPI app
        if FASTAPI_AVAILABLE:
            self.app = FastAPI(
                title="RF-DETR Surveillance API",
                description="High-performance person detection for surveillance applications",
                version="1.0.0"
            )
            self._setup_routes()
            self._setup_middleware()
        else:
            raise ImportError("FastAPI not available. Cannot create server.")
        
        logger.info("Inference server initialized")
    
    def _setup_middleware(self):
        """Setup middleware for the FastAPI app."""
        
        if self.config.enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.on_event("startup")
        async def startup_event():
            """Load model on startup."""
            success = self.model_manager.load_model()
            if not success:
                raise RuntimeError("Failed to load model on startup")
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "timestamp": time.time()}
        
        @self.app.get("/status", response_model=ServerStatus)
        async def get_status():
            """Get server status."""
            stats = self.metrics.get_stats()
            
            # Get memory usage
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            
            return ServerStatus(
                status="running",
                model_loaded=self.model_manager.model is not None,
                device=str(self.model_manager.device),
                model_type=self.config.model_type,
                uptime_seconds=stats['uptime_seconds'],
                requests_processed=stats['total_requests'],
                average_processing_time_ms=stats['average_processing_time_ms'],
                current_load=stats['current_load'],
                memory_usage_mb=memory_mb
            )
        
        @self.app.post("/detect", response_model=DetectionResult)
        async def detect_objects(request: DetectionRequest):
            """Detect objects in image."""
            request_id = request.request_id or str(uuid.uuid4())
            
            try:
                start_time = time.time()
                
                # Load image
                if request.image_data:
                    image = self._decode_base64_image(request.image_data)
                elif request.image_url:
                    image = await self._load_image_from_url(request.image_url)
                else:
                    raise HTTPException(status_code=400, detail="No image provided")
                
                # Run inference
                result = await asyncio.get_event_loop().run_in_executor(
                    self.executor, 
                    self._run_inference, 
                    image, 
                    request
                )
                
                # Convert to API response format
                detections = []
                detection_data = result['detections']
                
                if 'boxes' in detection_data and len(detection_data['boxes']) > 0:
                    boxes = detection_data['boxes'].cpu().numpy()
                    scores = detection_data['scores'].cpu().numpy()
                    labels = detection_data['labels'].cpu().numpy()
                    
                    # Map class IDs to names (simplified)
                    class_names = {1: 'person', 0: 'background'}  # Extend as needed
                    
                    for box, score, label in zip(boxes, scores, labels):
                        if score >= (request.confidence_threshold or self.config.confidence_threshold):
                            detections.append(BoundingBox(
                                x1=float(box[0]),
                                y1=float(box[1]),
                                x2=float(box[2]),
                                y2=float(box[3]),
                                confidence=float(score),
                                class_id=int(label),
                                class_name=class_names.get(int(label), f'class_{int(label)}')
                            ))
                
                processing_time_ms = (time.time() - start_time) * 1000
                
                # Record metrics
                self.metrics.add_request(processing_time_ms / 1000, success=True)
                
                return DetectionResult(
                    request_id=request_id,
                    detections=detections,
                    processing_time_ms=processing_time_ms,
                    image_shape=result['image_shape'],
                    model_info=result['model_info']
                )
                
            except Exception as e:
                self.metrics.add_request(0, success=False)
                logger.error(f"Detection failed for request {request_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/detect/upload")
        async def detect_upload(file: UploadFile = File(...)):
            """Detect objects in uploaded image file."""
            try:
                # Validate file
                if file.size > self.config.max_file_size_mb * 1024 * 1024:
                    raise HTTPException(status_code=413, detail="File too large")
                
                # Read and decode image
                content = await file.read()
                image = np.array(Image.open(io.BytesIO(content)))
                
                # Create request
                request = DetectionRequest()
                
                # Run inference
                result = await asyncio.get_event_loop().run_in_executor(
                    self.executor, 
                    self._run_inference, 
                    image, 
                    request
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Upload detection failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get performance metrics."""
            return self.metrics.get_stats()
    
    def _decode_base64_image(self, image_data: str) -> np.ndarray:
        """Decode base64 image data."""
        try:
            # Remove data URL prefix if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            
            # Convert to PIL Image then numpy
            image = Image.open(io.BytesIO(image_bytes))
            return np.array(image)
            
        except Exception as e:
            raise ValueError(f"Failed to decode image: {e}")
    
    async def _load_image_from_url(self, url: str) -> np.ndarray:
        """Load image from URL."""
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise ValueError(f"Failed to load image from URL: {response.status}")
                    
                    content = await response.read()
                    image = Image.open(io.BytesIO(content))
                    return np.array(image)
                    
        except ImportError:
            raise ValueError("aiohttp required for URL image loading")
        except Exception as e:
            raise ValueError(f"Failed to load image from URL: {e}")
    
    def _run_inference(self, image: np.ndarray, request: DetectionRequest) -> Dict[str, Any]:
        """Run model inference (synchronous)."""
        return self.model_manager.predict(image)
    
    def start_server(self):
        """Start the inference server."""
        logger.info(f"Starting inference server on {self.config.host}:{self.config.port}")
        
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            workers=self.config.workers,
            log_level=self.config.log_level.lower()
        )


def create_inference_server(
    model_path: Path,
    model_type: str = "torchscript",
    host: str = "0.0.0.0",
    port: int = 8000,
    device: str = "auto",
    **kwargs
) -> InferenceServer:
    """
    Create inference server with configuration.
    
    Args:
        model_path: Path to model file
        model_type: Type of model ("torchscript", "onnx", "tensorrt")
        host: Server host address
        port: Server port
        device: Inference device ("auto", "cpu", "cuda")
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured inference server
    """
    
    config = ServerConfig(
        model_path=model_path,
        model_type=model_type,
        host=host,
        port=port,
        device=device,
        **kwargs
    )
    
    return InferenceServer(config)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    if not FASTAPI_AVAILABLE:
        print("❌ FastAPI not available. Cannot run server.")
        exit(1)
    
    print("🧪 Testing Inference Server (Mock Mode)")
    
    # Create mock model for testing
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1, 91)
        
        def forward(self, x):
            batch_size = x.size(0)
            return {
                'pred_logits': torch.randn(batch_size, 100, 91),
                'pred_boxes': torch.rand(batch_size, 100, 4)
            }
    
    # Save mock model
    mock_model = MockModel()
    torch.jit.save(torch.jit.script(mock_model), "mock_model.pt")
    
    try:
        # Create server
        server = create_inference_server(
            model_path=Path("mock_model.pt"),
            model_type="torchscript",
            port=8000,
            enable_cors=True
        )
        
        print("✅ Inference server created")
        print("📋 API endpoints:")
        print("  GET  /health - Health check")
        print("  GET  /status - Server status")
        print("  POST /detect - Object detection")
        print("  POST /detect/upload - File upload detection")
        print("  GET  /metrics - Performance metrics")
        print("")
        print("🚀 To start server: server.start_server()")
        print("📝 API documentation available at: http://localhost:8000/docs")
        
        # Cleanup
        Path("mock_model.pt").unlink(missing_ok=True)
        
    except Exception as e:
        print(f"❌ Inference server test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("✅ Inference Server testing completed")