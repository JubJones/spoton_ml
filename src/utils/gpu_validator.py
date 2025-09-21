"""
GPU Validation and Memory Management for RF-DETR Training
"""
import logging
import torch
import psutil
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class GPUValidator:
    """Validates GPU capabilities and memory requirements for RF-DETR training."""
    
    # RF-DETR Memory Requirements (estimated in GB)
    MEMORY_REQUIREMENTS = {
        "rfdetr-nano": {"min": 4, "recommended": 6},
        "rfdetr-small": {"min": 6, "recommended": 8},
        "rfdetr-medium": {"min": 8, "recommended": 12},
        "rfdetr-large": {"min": 12, "recommended": 16},
        "rfdetr-base": {"min": 8, "recommended": 12},
    }
    
    def __init__(self):
        self.device_info = self._get_device_info()
    
    def _get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information."""
        info = {
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
            "cpu_count": psutil.cpu_count(),
            "system_memory_gb": psutil.virtual_memory().total / (1024**3),
        }
        
        if info["cuda_available"]:
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_devices"] = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                device_info = {
                    "id": i,
                    "name": props.name,
                    "memory_gb": props.total_memory / (1024**3),
                    "compute_capability": f"{props.major}.{props.minor}",
                }
                info["cuda_devices"].append(device_info)
        
        return info
    
    def validate_for_model(self, model_size: str, batch_size: int = 4) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validate if system can handle RF-DETR model training.
        
        Args:
            model_size: RF-DETR model size (e.g., "medium", "small")
            batch_size: Training batch size
            
        Returns:
            Tuple of (is_valid, recommendation_message, validation_details)
        """
        model_key = f"rfdetr-{model_size.lower()}"
        requirements = self.MEMORY_REQUIREMENTS.get(model_key, self.MEMORY_REQUIREMENTS["rfdetr-medium"])
        
        validation = {
            "model_size": model_size,
            "batch_size": batch_size,
            "requirements": requirements,
            "system_info": self.device_info,
            "recommendations": [],
            "warnings": [],
            "errors": [],
        }
        
        # Check CUDA availability and memory
        if self.device_info["cuda_available"]:
            best_gpu = max(self.device_info["cuda_devices"], key=lambda x: x["memory_gb"])
            validation["selected_device"] = f"cuda:{best_gpu['id']}"
            validation["available_memory_gb"] = best_gpu["memory_gb"]
            
            # Memory validation with batch size scaling
            estimated_memory_needed = requirements["min"] * (batch_size / 4.0)  # Base estimate for batch_size=4
            
            if best_gpu["memory_gb"] >= requirements["recommended"] * (batch_size / 4.0):
                validation["memory_status"] = "optimal"
                validation["recommendations"].append(f"✅ Excellent GPU memory ({best_gpu['memory_gb']:.1f}GB) for RF-DETR {model_size}")
            elif best_gpu["memory_gb"] >= estimated_memory_needed:
                validation["memory_status"] = "sufficient"
                validation["recommendations"].append(f"⚠️  Sufficient GPU memory, consider reducing batch size if OOM occurs")
            else:
                validation["memory_status"] = "insufficient"
                validation["errors"].append(f"❌ Insufficient GPU memory: {best_gpu['memory_gb']:.1f}GB < {estimated_memory_needed:.1f}GB needed")
                
                # Suggest alternatives
                max_batch_size = int(4 * (best_gpu["memory_gb"] / requirements["min"]))
                if max_batch_size >= 1:
                    validation["recommendations"].append(f"📉 Reduce batch_size to {max_batch_size} or enable gradient accumulation")
                else:
                    validation["recommendations"].append("📱 Consider using RF-DETR Small or Nano model")
                    
        elif self.device_info["mps_available"]:
            validation["selected_device"] = "mps"
            validation["memory_status"] = "unknown"
            validation["warnings"].append("⚠️  MPS device detected - memory estimation not available")
            validation["recommendations"].append("🍎 Monitor memory usage during training on Apple Silicon")
            
        else:
            validation["selected_device"] = "cpu"
            validation["memory_status"] = "cpu_fallback"
            validation["warnings"].append("⚠️  No GPU detected - training will be significantly slower on CPU")
            validation["recommendations"].append("🐌 Consider cloud GPU services (Colab, AWS, etc.) for faster training")
        
        # Additional recommendations
        if batch_size < 4:
            validation["recommendations"].append("📊 Small batch size detected - consider gradient accumulation for stable training")
        if batch_size > 8 and validation.get("available_memory_gb", 0) < 16:
            validation["warnings"].append("⚠️  Large batch size with limited memory - monitor for OOM errors")
            
        # Overall validation status
        is_valid = len(validation["errors"]) == 0
        
        # Generate summary message
        if is_valid:
            if validation["memory_status"] == "optimal":
                message = f"✅ System validated for RF-DETR {model_size} training with optimal performance"
            else:
                message = f"✅ System can handle RF-DETR {model_size} training with some limitations"
        else:
            message = f"❌ System validation failed for RF-DETR {model_size} training"
            
        return is_valid, message, validation
    
    def get_optimal_batch_size(self, model_size: str, target_memory_usage: float = 0.8) -> int:
        """
        Calculate optimal batch size based on available GPU memory.
        
        Args:
            model_size: RF-DETR model size
            target_memory_usage: Target GPU memory usage (0.0-1.0)
            
        Returns:
            Recommended batch size
        """
        if not self.device_info["cuda_available"]:
            return 2  # Conservative batch size for non-CUDA devices
            
        best_gpu = max(self.device_info["cuda_devices"], key=lambda x: x["memory_gb"])
        available_memory = best_gpu["memory_gb"] * target_memory_usage
        
        model_key = f"rfdetr-{model_size.lower()}"
        base_memory = self.MEMORY_REQUIREMENTS.get(model_key, self.MEMORY_REQUIREMENTS["rfdetr-medium"])["min"]
        
        # Estimate batch size (linear scaling assumption)
        optimal_batch_size = max(1, int(4 * (available_memory / base_memory)))
        
        # Cap at reasonable limits
        return min(optimal_batch_size, 16)
    
    def log_system_info(self):
        """Log comprehensive system information."""
        logger.info("=== System Validation Report ===")
        logger.info(f"CPU cores: {self.device_info['cpu_count']}")
        logger.info(f"System RAM: {self.device_info['system_memory_gb']:.1f}GB")
        
        if self.device_info["cuda_available"]:
            logger.info(f"CUDA devices: {self.device_info['cuda_device_count']}")
            for device in self.device_info["cuda_devices"]:
                logger.info(f"  GPU {device['id']}: {device['name']} ({device['memory_gb']:.1f}GB)")
        elif self.device_info["mps_available"]:
            logger.info("MPS (Apple Silicon) available")
        else:
            logger.info("No GPU acceleration available")


def validate_training_environment(model_size: str = "medium", batch_size: int = 4) -> Dict[str, Any]:
    """
    Comprehensive training environment validation.
    
    Args:
        model_size: RF-DETR model size
        batch_size: Planned batch size
        
    Returns:
        Validation results dictionary
    """
    validator = GPUValidator()
    validator.log_system_info()
    
    is_valid, message, details = validator.validate_for_model(model_size, batch_size)
    
    logger.info(f"\n{message}")
    
    for rec in details["recommendations"]:
        logger.info(rec)
    for warning in details["warnings"]:
        logger.warning(warning)
    for error in details["errors"]:
        logger.error(error)
    
    if is_valid:
        optimal_batch = validator.get_optimal_batch_size(model_size)
        if optimal_batch != batch_size:
            logger.info(f"💡 Optimal batch size recommendation: {optimal_batch}")
            details["optimal_batch_size"] = optimal_batch
    
    return details


if __name__ == "__main__":
    # Example validation
    logging.basicConfig(level=logging.INFO)
    result = validate_training_environment("medium", 4)
    print(f"\nValidation complete. Status: {'✅ PASSED' if len(result['errors']) == 0 else '❌ FAILED'}")