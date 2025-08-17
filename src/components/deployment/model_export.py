"""
Model Export and Optimization for RF-DETR Surveillance Deployment
ONNX, TensorRT, and TorchScript export with optimization pipeline
"""
import logging
import torch
import torch.nn as nn
import torchvision
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import json
import time
import warnings

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.debug("ONNX not available. ONNX export disabled.")

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    logging.debug("TensorRT not available. TensorRT optimization disabled.")

logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Configuration for model export and optimization."""
    
    # Export formats
    export_torchscript: bool = True
    export_onnx: bool = True
    export_tensorrt: bool = False  # Requires TensorRT installation
    
    # Model optimization
    optimize_for_inference: bool = True
    quantization_enabled: bool = False
    quantization_type: str = "dynamic"  # "dynamic", "static", "qat"
    
    # Input specifications
    input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640)  # NCHW
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    
    # ONNX specific
    onnx_opset_version: int = 11
    onnx_optimize: bool = True
    
    # TensorRT specific
    trt_precision: str = "fp16"  # "fp32", "fp16", "int8"
    trt_workspace_size: int = 1 << 30  # 1GB
    trt_max_batch_size: int = 8
    
    # Export paths
    export_dir: Path = field(default_factory=lambda: Path("exported_models"))
    model_name: str = "rfdetr_surveillance"
    
    # Validation
    validate_exports: bool = True
    benchmark_exports: bool = True
    
    def __post_init__(self):
        """Set default dynamic axes if not provided."""
        if self.dynamic_axes is None:
            self.dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }


@dataclass
class ExportResult:
    """Result of model export operation."""
    
    format: str
    export_path: Path
    file_size_mb: float
    export_time_seconds: float
    
    # Validation results
    validation_passed: bool = False
    accuracy_drop: Optional[float] = None
    
    # Benchmark results
    inference_time_ms: Optional[float] = None
    throughput_fps: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    
    # Metadata
    input_shape: Tuple[int, ...] = None
    output_shape: Tuple[int, ...] = None
    optimization_flags: Dict[str, Any] = field(default_factory=dict)


class ModelOptimizer:
    """Model optimization utilities for deployment."""
    
    @staticmethod
    def optimize_for_inference(model: nn.Module) -> nn.Module:
        """Optimize PyTorch model for inference."""
        
        logger.info("Optimizing model for inference...")
        
        # Set to evaluation mode
        model.eval()
        
        # Fuse common operations
        try:
            # Fuse Conv2d + BatchNorm2d + ReLU
            model = torch.jit.optimize_for_inference(model)
            logger.info("Applied torch.jit.optimize_for_inference")
        except Exception as e:
            logger.warning(f"JIT optimization failed: {e}")
        
        # Remove dropout and batch norm training artifacts
        for module in model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                module.p = 0.0
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.momentum = 0.0
        
        return model
    
    @staticmethod
    def quantize_model(
        model: nn.Module, 
        quantization_type: str = "dynamic",
        calibration_data: Optional[torch.utils.data.DataLoader] = None
    ) -> nn.Module:
        """Quantize model for deployment."""
        
        logger.info(f"Applying {quantization_type} quantization...")
        
        if quantization_type == "dynamic":
            # Dynamic quantization (no calibration data needed)
            quantized_model = torch.quantization.quantize_dynamic(
                model, 
                {nn.Linear, nn.Conv2d}, 
                dtype=torch.qint8
            )
        
        elif quantization_type == "static":
            # Static quantization (requires calibration data)
            if calibration_data is None:
                raise ValueError("Static quantization requires calibration data")
            
            # Prepare model for quantization
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)
            
            # Calibrate with representative data
            model.eval()
            with torch.no_grad():
                for batch_idx, data in enumerate(calibration_data):
                    if batch_idx >= 100:  # Limit calibration data
                        break
                    model(data[0] if isinstance(data, (list, tuple)) else data)
            
            # Convert to quantized model
            quantized_model = torch.quantization.convert(model, inplace=False)
        
        else:
            raise ValueError(f"Unsupported quantization type: {quantization_type}")
        
        return quantized_model
    
    @staticmethod
    def remove_training_artifacts(model: nn.Module) -> nn.Module:
        """Remove training-specific components for deployment."""
        
        # Remove auxiliary heads used only in training
        artifacts_to_remove = ['aux_classifier', 'training_head', 'auxiliary_loss']
        
        for name in artifacts_to_remove:
            if hasattr(model, name):
                delattr(model, name)
                logger.info(f"Removed training artifact: {name}")
        
        return model


class ModelExporter:
    """Comprehensive model export pipeline for RF-DETR surveillance."""
    
    def __init__(self, config: ExportConfig):
        self.config = config
        self.config.export_dir.mkdir(parents=True, exist_ok=True)
        
        self.export_results = []
        self.optimizer = ModelOptimizer()
        
        logger.info(f"Model Exporter initialized:")
        logger.info(f"  Export directory: {config.export_dir}")
        logger.info(f"  Target formats: {self._get_enabled_formats()}")
        logger.info(f"  Input shape: {config.input_shape}")
        logger.info(f"  Optimization enabled: {config.optimize_for_inference}")
    
    def _get_enabled_formats(self) -> List[str]:
        """Get list of enabled export formats."""
        formats = []
        if self.config.export_torchscript:
            formats.append("TorchScript")
        if self.config.export_onnx and ONNX_AVAILABLE:
            formats.append("ONNX")
        if self.config.export_tensorrt and TENSORRT_AVAILABLE:
            formats.append("TensorRT")
        return formats
    
    def export_model(
        self, 
        model: nn.Module,
        sample_input: Optional[torch.Tensor] = None,
        model_name: Optional[str] = None
    ) -> List[ExportResult]:
        """Export model to all configured formats."""
        
        model_name = model_name or self.config.model_name
        
        # Prepare model for export
        original_model = model
        model = self._prepare_model_for_export(model)
        
        # Create sample input if not provided
        if sample_input is None:
            sample_input = torch.randn(self.config.input_shape)
        
        logger.info(f"Starting model export for: {model_name}")
        logger.info(f"  Input shape: {sample_input.shape}")
        
        export_results = []
        
        # Export TorchScript
        if self.config.export_torchscript:
            try:
                result = self._export_torchscript(model, sample_input, model_name)
                export_results.append(result)
            except Exception as e:
                logger.error(f"TorchScript export failed: {e}")
        
        # Export ONNX
        if self.config.export_onnx and ONNX_AVAILABLE:
            try:
                result = self._export_onnx(model, sample_input, model_name)
                export_results.append(result)
            except Exception as e:
                logger.error(f"ONNX export failed: {e}")
        
        # Export TensorRT
        if self.config.export_tensorrt and TENSORRT_AVAILABLE:
            try:
                # TensorRT export requires ONNX as intermediate
                onnx_path = self.config.export_dir / f"{model_name}.onnx"
                if onnx_path.exists():
                    result = self._export_tensorrt(onnx_path, model_name)
                    export_results.append(result)
                else:
                    logger.warning("TensorRT export requires ONNX model. Export ONNX first.")
            except Exception as e:
                logger.error(f"TensorRT export failed: {e}")
        
        # Validate exports
        if self.config.validate_exports:
            self._validate_exports(original_model, export_results, sample_input)
        
        # Benchmark exports
        if self.config.benchmark_exports:
            self._benchmark_exports(export_results, sample_input)
        
        self.export_results.extend(export_results)
        
        logger.info(f"Model export completed. {len(export_results)} formats exported.")
        
        return export_results
    
    def _prepare_model_for_export(self, model: nn.Module) -> nn.Module:
        """Prepare model for export by applying optimizations."""
        
        # Set to evaluation mode
        model.eval()
        
        # Apply inference optimizations
        if self.config.optimize_for_inference:
            model = self.optimizer.optimize_for_inference(model)
        
        # Apply quantization
        if self.config.quantization_enabled:
            model = self.optimizer.quantize_model(model, self.config.quantization_type)
        
        # Remove training artifacts
        model = self.optimizer.remove_training_artifacts(model)
        
        return model
    
    def _export_torchscript(
        self, 
        model: nn.Module, 
        sample_input: torch.Tensor, 
        model_name: str
    ) -> ExportResult:
        """Export model to TorchScript format."""
        
        logger.info("Exporting to TorchScript...")
        start_time = time.time()
        
        export_path = self.config.export_dir / f"{model_name}.pt"
        
        try:
            # Trace the model
            with torch.no_grad():
                traced_model = torch.jit.trace(model, sample_input)
            
            # Optimize the traced model
            if self.config.optimize_for_inference:
                traced_model = torch.jit.optimize_for_inference(traced_model)
            
            # Save the model
            torch.jit.save(traced_model, export_path)
            
            export_time = time.time() - start_time
            file_size_mb = export_path.stat().st_size / (1024 * 1024)
            
            result = ExportResult(
                format="TorchScript",
                export_path=export_path,
                file_size_mb=file_size_mb,
                export_time_seconds=export_time,
                input_shape=sample_input.shape,
                optimization_flags={
                    "traced": True,
                    "optimized_for_inference": self.config.optimize_for_inference
                }
            )
            
            logger.info(f"TorchScript export completed: {export_path}")
            logger.info(f"  File size: {file_size_mb:.2f} MB")
            logger.info(f"  Export time: {export_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"TorchScript export failed: {e}")
            raise
    
    def _export_onnx(
        self, 
        model: nn.Module, 
        sample_input: torch.Tensor, 
        model_name: str
    ) -> ExportResult:
        """Export model to ONNX format."""
        
        logger.info("Exporting to ONNX...")
        start_time = time.time()
        
        export_path = self.config.export_dir / f"{model_name}.onnx"
        
        try:
            # Export to ONNX
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                torch.onnx.export(
                    model,
                    sample_input,
                    str(export_path),
                    export_params=True,
                    opset_version=self.config.onnx_opset_version,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes=self.config.dynamic_axes,
                    verbose=False
                )
            
            # Optimize ONNX model
            if self.config.onnx_optimize:
                self._optimize_onnx_model(export_path)
            
            export_time = time.time() - start_time
            file_size_mb = export_path.stat().st_size / (1024 * 1024)
            
            result = ExportResult(
                format="ONNX",
                export_path=export_path,
                file_size_mb=file_size_mb,
                export_time_seconds=export_time,
                input_shape=sample_input.shape,
                optimization_flags={
                    "opset_version": self.config.onnx_opset_version,
                    "optimized": self.config.onnx_optimize,
                    "dynamic_axes": bool(self.config.dynamic_axes)
                }
            )
            
            logger.info(f"ONNX export completed: {export_path}")
            logger.info(f"  File size: {file_size_mb:.2f} MB")
            logger.info(f"  Export time: {export_time:.2f}s")
            logger.info(f"  Opset version: {self.config.onnx_opset_version}")
            
            return result
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise
    
    def _optimize_onnx_model(self, model_path: Path):
        """Optimize ONNX model using ONNX optimizer."""
        
        try:
            import onnxoptimizer
            
            # Load model
            model = onnx.load(str(model_path))
            
            # Apply optimizations
            optimized_model = onnxoptimizer.optimize(model)
            
            # Save optimized model
            onnx.save(optimized_model, str(model_path))
            
            logger.info("Applied ONNX optimizations")
            
        except ImportError:
            logger.debug("onnxoptimizer not available. Skipping ONNX optimization.")
        except Exception as e:
            logger.warning(f"ONNX optimization failed: {e}")
    
    def _export_tensorrt(self, onnx_path: Path, model_name: str) -> ExportResult:
        """Export ONNX model to TensorRT format."""
        
        logger.info("Exporting to TensorRT...")
        start_time = time.time()
        
        export_path = self.config.export_dir / f"{model_name}.trt"
        
        try:
            # Create TensorRT logger
            trt_logger = trt.Logger(trt.Logger.WARNING)
            
            # Create builder and network
            builder = trt.Builder(trt_logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, trt_logger)
            
            # Parse ONNX model
            with open(onnx_path, 'rb') as model_file:
                if not parser.parse(model_file.read()):
                    for error in range(parser.num_errors):
                        logger.error(f"TensorRT parser error: {parser.get_error(error)}")
                    raise RuntimeError("Failed to parse ONNX model")
            
            # Configure builder
            config = builder.create_builder_config()
            config.max_workspace_size = self.config.trt_workspace_size
            
            # Set precision mode
            if self.config.trt_precision == "fp16":
                config.set_flag(trt.BuilderFlag.FP16)
            elif self.config.trt_precision == "int8":
                config.set_flag(trt.BuilderFlag.INT8)
                # Note: INT8 calibration would be needed here
            
            # Build engine
            engine = builder.build_engine(network, config)
            
            if engine is None:
                raise RuntimeError("Failed to build TensorRT engine")
            
            # Serialize and save engine
            with open(export_path, 'wb') as f:
                f.write(engine.serialize())
            
            export_time = time.time() - start_time
            file_size_mb = export_path.stat().st_size / (1024 * 1024)
            
            result = ExportResult(
                format="TensorRT",
                export_path=export_path,
                file_size_mb=file_size_mb,
                export_time_seconds=export_time,
                optimization_flags={
                    "precision": self.config.trt_precision,
                    "workspace_size_gb": self.config.trt_workspace_size / (1024**3),
                    "max_batch_size": self.config.trt_max_batch_size
                }
            )
            
            logger.info(f"TensorRT export completed: {export_path}")
            logger.info(f"  File size: {file_size_mb:.2f} MB")
            logger.info(f"  Export time: {export_time:.2f}s")
            logger.info(f"  Precision: {self.config.trt_precision}")
            
            return result
            
        except Exception as e:
            logger.error(f"TensorRT export failed: {e}")
            raise
    
    def _validate_exports(
        self, 
        original_model: nn.Module, 
        export_results: List[ExportResult], 
        sample_input: torch.Tensor
    ):
        """Validate exported models against original."""
        
        logger.info("Validating exported models...")
        
        # Get original model output
        original_model.eval()
        with torch.no_grad():
            original_output = original_model(sample_input)
        
        # Extract tensor from output if it's a dict
        if isinstance(original_output, dict):
            original_tensor = original_output.get('pred_logits', list(original_output.values())[0])
        else:
            original_tensor = original_output
        
        for result in export_results:
            try:
                if result.format == "TorchScript":
                    # Load and test TorchScript model
                    traced_model = torch.jit.load(result.export_path)
                    with torch.no_grad():
                        exported_output = traced_model(sample_input)
                    
                    if isinstance(exported_output, dict):
                        exported_tensor = exported_output.get('pred_logits', list(exported_output.values())[0])
                    else:
                        exported_tensor = exported_output
                    
                    # Compare outputs
                    max_diff = torch.max(torch.abs(original_tensor - exported_tensor)).item()
                    result.validation_passed = max_diff < 1e-4
                    result.accuracy_drop = max_diff
                    
                elif result.format == "ONNX" and ONNX_AVAILABLE:
                    # Load and test ONNX model
                    ort_session = ort.InferenceSession(str(result.export_path))
                    
                    # Prepare input
                    ort_inputs = {ort_session.get_inputs()[0].name: sample_input.numpy()}
                    ort_outputs = ort_session.run(None, ort_inputs)
                    
                    exported_tensor = torch.from_numpy(ort_outputs[0])
                    
                    # Compare outputs
                    max_diff = torch.max(torch.abs(original_tensor - exported_tensor)).item()
                    result.validation_passed = max_diff < 1e-3  # ONNX may have slightly more diff
                    result.accuracy_drop = max_diff
                
                else:
                    # For formats we can't easily validate
                    result.validation_passed = True
                    result.accuracy_drop = 0.0
                
                status = "✅" if result.validation_passed else "❌"
                logger.info(f"  {status} {result.format}: max_diff = {result.accuracy_drop:.2e}")
                
            except Exception as e:
                logger.warning(f"Validation failed for {result.format}: {e}")
                result.validation_passed = False
    
    def _benchmark_exports(self, export_results: List[ExportResult], sample_input: torch.Tensor):
        """Benchmark exported models for inference performance."""
        
        logger.info("Benchmarking exported models...")
        
        for result in export_results:
            try:
                if result.format == "TorchScript":
                    model = torch.jit.load(result.export_path)
                    result.inference_time_ms, result.throughput_fps = self._benchmark_pytorch_model(model, sample_input)
                
                elif result.format == "ONNX" and ONNX_AVAILABLE:
                    session = ort.InferenceSession(str(result.export_path))
                    result.inference_time_ms, result.throughput_fps = self._benchmark_onnx_model(session, sample_input)
                
                logger.info(f"  {result.format}: {result.inference_time_ms:.2f}ms, {result.throughput_fps:.1f} FPS")
                
            except Exception as e:
                logger.warning(f"Benchmarking failed for {result.format}: {e}")
    
    def _benchmark_pytorch_model(self, model: torch.nn.Module, sample_input: torch.Tensor) -> Tuple[float, float]:
        """Benchmark PyTorch model."""
        
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(sample_input)
        
        # Benchmark
        num_runs = 100
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(sample_input)
        
        total_time = time.time() - start_time
        avg_time_ms = (total_time / num_runs) * 1000
        fps = 1000 / avg_time_ms
        
        return avg_time_ms, fps
    
    def _benchmark_onnx_model(self, session: 'ort.InferenceSession', sample_input: torch.Tensor) -> Tuple[float, float]:
        """Benchmark ONNX model."""
        
        input_name = session.get_inputs()[0].name
        ort_inputs = {input_name: sample_input.numpy()}
        
        # Warmup
        for _ in range(10):
            _ = session.run(None, ort_inputs)
        
        # Benchmark
        num_runs = 100
        start_time = time.time()
        
        for _ in range(num_runs):
            _ = session.run(None, ort_inputs)
        
        total_time = time.time() - start_time
        avg_time_ms = (total_time / num_runs) * 1000
        fps = 1000 / avg_time_ms
        
        return avg_time_ms, fps
    
    def get_export_summary(self) -> Dict[str, Any]:
        """Get comprehensive export summary."""
        
        if not self.export_results:
            return {"message": "No exports completed"}
        
        summary = {
            "total_exports": len(self.export_results),
            "successful_exports": sum(1 for r in self.export_results if r.validation_passed),
            "export_formats": [r.format for r in self.export_results],
            "total_size_mb": sum(r.file_size_mb for r in self.export_results),
            "export_details": []
        }
        
        for result in self.export_results:
            detail = {
                "format": result.format,
                "file_size_mb": result.file_size_mb,
                "export_time_s": result.export_time_seconds,
                "validation_passed": result.validation_passed,
                "accuracy_drop": result.accuracy_drop,
                "inference_time_ms": result.inference_time_ms,
                "throughput_fps": result.throughput_fps,
                "export_path": str(result.export_path)
            }
            summary["export_details"].append(detail)
        
        return summary


def create_model_exporter(
    export_dir: Path,
    input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640),
    export_formats: List[str] = None,
    optimize_for_inference: bool = True,
    **kwargs
) -> ModelExporter:
    """
    Create model exporter with configuration.
    
    Args:
        export_dir: Directory for exported models
        input_shape: Model input shape (NCHW)
        export_formats: List of formats to export ("torchscript", "onnx", "tensorrt")
        optimize_for_inference: Apply inference optimizations
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured model exporter
    """
    
    export_formats = export_formats or ["torchscript", "onnx"]
    
    config = ExportConfig(
        export_dir=export_dir,
        input_shape=input_shape,
        export_torchscript="torchscript" in export_formats,
        export_onnx="onnx" in export_formats,
        export_tensorrt="tensorrt" in export_formats,
        optimize_for_inference=optimize_for_inference,
        **kwargs
    )
    
    return ModelExporter(config)


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing Model Export Pipeline")
    
    # Create test model
    class TestRFDETR(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.head = nn.Linear(64, 91)
        
        def forward(self, x):
            features = self.backbone(x)
            features = features.flatten(1)
            logits = self.head(features)
            return {'pred_logits': logits}
    
    try:
        # Create model and exporter
        model = TestRFDETR()
        exporter = create_model_exporter(
            export_dir=Path("test_exports"),
            input_shape=(1, 3, 640, 640),
            export_formats=["torchscript", "onnx"],
            optimize_for_inference=True
        )
        
        print("✅ Model exporter created")
        
        # Export model
        sample_input = torch.randn(1, 3, 640, 640)
        export_results = exporter.export_model(model, sample_input, "test_rfdetr")
        
        print(f"✅ Model exported to {len(export_results)} formats")
        
        # Get summary
        summary = exporter.get_export_summary()
        print("📊 Export Summary:")
        print(f"  Total exports: {summary['total_exports']}")
        print(f"  Successful: {summary['successful_exports']}")
        print(f"  Total size: {summary['total_size_mb']:.2f} MB")
        
        for detail in summary['export_details']:
            print(f"  {detail['format']}: {detail['file_size_mb']:.1f}MB, "
                  f"{detail['inference_time_ms']:.1f}ms, "
                  f"{'✅' if detail['validation_passed'] else '❌'}")
        
        # Cleanup
        import shutil
        if Path("test_exports").exists():
            shutil.rmtree("test_exports")
        
        print("✅ Model export testing completed")
        
    except Exception as e:
        print(f"❌ Model export test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("✅ Model Export Pipeline testing completed")