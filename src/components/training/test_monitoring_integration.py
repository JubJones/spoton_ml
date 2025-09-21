#!/usr/bin/env python3
"""
Comprehensive Integration Test for RF-DETR Training Monitoring Infrastructure
Tests all Wave 3 components: EMA, Loss Tracking, and Comprehensive Monitoring
"""
import logging
import sys
from pathlib import Path
import torch
import torch.nn as nn
import tempfile
import shutil

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.components.training.monitoring import (
    EMAHandler, EMAConfig, 
    LossTracker, LossConfig,
    TrainingMonitor, MonitorConfig,
    MetricsLogger, LoggerConfig,
    create_ema_handler, create_loss_tracker,
    create_training_monitor, create_metrics_logger
)
from src.components.training.optimizers import create_rfdetr_optimizer
from src.components.training.schedulers import create_cosine_scheduler_with_warmup

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockRFDETRModel(nn.Module):
    """Mock RF-DETR model for testing."""
    
    def __init__(self):
        super().__init__()
        # Simulate RF-DETR architecture components
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        self.neck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 64, 256),
            nn.ReLU()
        )
        
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 91)  # COCO classes
        )
        
        # Add some transformer-like components
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(256, 8, batch_first=True),
            num_layers=2
        )
    
    def forward(self, x):
        # Simulate RF-DETR forward pass
        features = self.backbone(x)
        neck_features = self.neck(features)
        
        # Simulate transformer processing
        transformer_out = self.encoder(neck_features.unsqueeze(1))
        
        # Generate outputs
        outputs = self.head(transformer_out.squeeze(1))
        
        return {
            'pred_logits': outputs[:, :91],
            'pred_boxes': torch.rand(x.size(0), 4)  # Mock box predictions
        }


def mock_validation_function(model):
    """Mock validation function that returns metrics."""
    model.eval()
    
    # Simulate validation metrics
    with torch.no_grad():
        # Simulate some validation batches
        total_loss = 0
        accuracy = 0
        
        for i in range(5):  # 5 validation batches
            # Mock validation computation
            batch_loss = 0.5 + torch.rand(1).item() * 0.3
            batch_acc = 0.7 + torch.rand(1).item() * 0.25
            
            total_loss += batch_loss
            accuracy += batch_acc
        
        avg_loss = total_loss / 5
        avg_accuracy = accuracy / 5
    
    return {
        'val_loss': avg_loss,
        'val_accuracy': avg_accuracy,
        'val_map': avg_accuracy * 0.8,  # Mock mAP
        'val_precision': avg_accuracy * 0.9,
        'val_recall': avg_accuracy * 0.85
    }


def test_ema_handler():
    """Test EMA handler functionality."""
    logger.info("Testing EMA Handler...")
    
    model = MockRFDETRModel()
    
    # Test with different configurations
    configs = [
        EMAConfig(decay=0.999, tau=1000, warmup_steps=10),
        EMAConfig(decay=0.9999, tau=2000, dynamic_decay=True, decay_schedule="cosine"),
        EMAConfig(decay=0.998, momentum_based=True, warmup_steps=5)
    ]
    
    for i, config in enumerate(configs):
        logger.info(f"  Testing EMA config {i+1}: decay={config.decay}, schedule={config.decay_schedule}")
        
        ema_handler = EMAHandler(model, config)
        
        # Simulate training steps
        for step in range(50):
            # Simulate parameter updates (modify model weights)
            with torch.no_grad():
                for param in model.parameters():
                    param.data += torch.randn_like(param.data) * 0.001
            
            ema_handler.update(step)
        
        # Test validation comparison
        comparison = ema_handler.validate_ema_performance(mock_validation_function)
        
        logger.info(f"    EMA validation: {comparison['ema_better_count']}/{comparison['total_metrics']} metrics improved")
        
        # Test statistics
        stats = ema_handler.get_ema_statistics()
        assert stats['step_count'] == 49
        assert stats['update_count'] > 0
        
        # Test state saving/loading
        state = ema_handler.save_state()
        new_ema = EMAHandler(model, config)
        new_ema.load_state(state)
        
        assert new_ema.step_count == ema_handler.step_count
        assert new_ema.update_count == ema_handler.update_count
    
    logger.info("✅ EMA Handler tests passed")


def test_loss_tracker():
    """Test loss tracker functionality."""
    logger.info("Testing Loss Tracker...")
    
    # Test with different configurations
    configs = [
        LossConfig(window_size=100, smooth_window=20, log_frequency=10),
        LossConfig(window_size=500, anomaly_detection=True, correlation_analysis=True),
        LossConfig(window_size=200, convergence_analysis=True, trend_analysis_window=50)
    ]
    
    for i, config in enumerate(configs):
        logger.info(f"  Testing Loss Tracker config {i+1}")
        
        tracker = LossTracker(config)
        
        # Simulate training with decreasing loss (convergence)
        base_loss = 2.0
        for step in range(100):
            # Simulate RF-DETR loss components
            progress = step / 100.0
            noise = torch.randn(1).item() * 0.05
            
            loss_dict = {
                'loss': base_loss * (1 - progress * 0.8) + noise,
                'loss_ce': base_loss * 0.6 * (1 - progress * 0.7) + noise * 0.5,
                'loss_bbox': base_loss * 0.3 * (1 - progress * 0.9) + noise * 0.3,
                'loss_giou': base_loss * 0.1 * (1 - progress * 0.85) + noise * 0.2,
                'class_error': 50 * (1 - progress * 0.6) + abs(noise) * 5
            }
            
            tracker.update(loss_dict, step=step)
            
            # Inject some anomalies for testing
            if step == 30:
                anomaly_loss = {'loss': base_loss * 3.0, 'loss_ce': base_loss * 2.0}
                tracker.update(anomaly_loss, step=step)
            
        # Test summary and analysis
        summary = tracker.get_loss_summary()
        assert summary['step_count'] == 100
        assert 'total_loss_stats' in summary
        assert 'convergence_metrics' in summary
        
        # Test anomaly detection
        anomaly_report = tracker.get_anomaly_report()
        if config.anomaly_detection:
            logger.info(f"    Detected {anomaly_report['total_anomalies']} anomalies")
        
        # Test correlation analysis
        if config.correlation_analysis:
            correlations = tracker.analyze_loss_correlation()
            logger.info(f"    Found {len(correlations)} loss correlations")
        
        # Test convergence analysis
        if config.convergence_analysis:
            conv_metrics = summary['convergence_metrics']
            logger.info(f"    Converging: {conv_metrics['is_converging']}, "
                       f"Stability: {conv_metrics['loss_stability']:.3f}")
    
    logger.info("✅ Loss Tracker tests passed")


def test_metrics_logger():
    """Test metrics logger functionality."""
    logger.info("Testing Metrics Logger...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test different logger configurations
        configs = [
            LoggerConfig(json_logging=True, csv_logging=False, async_logging=False),
            LoggerConfig(json_logging=True, csv_logging=True, async_logging=False),
            LoggerConfig(json_logging=True, csv_logging=True, async_logging=True, flush_frequency=10)
        ]
        
        for i, config in enumerate(configs):
            logger.info(f"  Testing Metrics Logger config {i+1}")
            
            log_dir = Path(temp_dir) / f"test_logs_{i}"
            metrics_logger = MetricsLogger(str(log_dir), config)
            
            # Log various types of metrics
            for step in range(50):
                metrics = {
                    'loss': {
                        'total': 1.0 - step * 0.01,
                        'components': {
                            'ce': 0.6 - step * 0.005,
                            'bbox': 0.3 - step * 0.003,
                            'giou': 0.1 - step * 0.001
                        }
                    },
                    'performance': {
                        'accuracy': 0.5 + step * 0.008,
                        'precision': 0.4 + step * 0.009,
                        'recall': 0.3 + step * 0.01
                    },
                    'system': {
                        'memory_mb': 1000 + step * 5,
                        'step_time': 0.1 + step * 0.001
                    }
                }
                
                metrics_logger.log_metrics(metrics, step=step)
                
                # Log some events
                if step % 10 == 0:
                    metrics_logger.log_event("checkpoint", {
                        "step": step,
                        "model_size": f"{100 + step}MB"
                    })
            
            # Test statistics
            stats = metrics_logger.get_statistics()
            assert stats['total_entries'] >= 50
            
            # Force flush and close
            metrics_logger.flush()
            metrics_logger.close()
            
            # Verify log files were created
            if config.json_logging:
                json_files = list((log_dir / "json").glob("*.jsonl"))
                assert len(json_files) > 0, "JSON log files should be created"
            
            if config.csv_logging:
                csv_files = list((log_dir / "csv").glob("*.csv"))
                assert len(csv_files) > 0, "CSV log files should be created"
    
    logger.info("✅ Metrics Logger tests passed")


def test_training_monitor_integration():
    """Test comprehensive training monitor integration."""
    logger.info("Testing Training Monitor Integration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        model = MockRFDETRModel()
        
        # Create comprehensive monitoring configuration
        monitor_config = MonitorConfig(
            ema_enabled=True,
            loss_tracking_enabled=True,
            metrics_logging_enabled=True,
            validation_enabled=True,
            validation_frequency=20,
            checkpoint_enabled=True,
            checkpoint_frequency=30,
            performance_monitoring=True,
            memory_monitoring=True
        )
        
        monitor = TrainingMonitor(model, monitor_config, save_dir=temp_dir)
        
        # Simulate comprehensive training workflow
        with monitor:
            # Simulate multiple epochs
            for epoch in range(3):
                monitor.start_epoch(epoch)
                
                for step in range(40):
                    global_step = epoch * 40 + step
                    
                    # Simulate RF-DETR loss components with decreasing trend
                    progress = global_step / 120.0
                    base_loss = 2.5 * (1 - progress * 0.7)
                    noise = torch.randn(1).item() * 0.03
                    
                    loss_dict = {
                        'loss': base_loss + noise,
                        'loss_ce': base_loss * 0.6 + noise * 0.5,
                        'loss_bbox': base_loss * 0.25 + noise * 0.3,
                        'loss_giou': base_loss * 0.15 + noise * 0.2,
                        'class_error': 30 * (1 - progress * 0.5) + abs(noise) * 3,
                        'cardinality_error': 5 * (1 - progress * 0.8) + abs(noise)
                    }
                    
                    # Additional metrics
                    additional_metrics = {
                        'learning_rate': 1e-4 * (0.99 ** global_step),
                        'gradient_norm': 1.0 + torch.randn(1).item() * 0.2,
                        'batch_size': 32
                    }
                    
                    monitor.step(
                        loss_dict, 
                        step=global_step, 
                        epoch=epoch,
                        additional_metrics=additional_metrics
                    )
                    
                    # Periodic validation
                    if global_step % 20 == 0 and global_step > 0:
                        validation_metrics = {'epoch': epoch, 'global_step': global_step}
                        validation_results = monitor.validate(
                            mock_validation_function,
                            validation_metrics=validation_metrics
                        )
                        
                        logger.info(f"    Validation at step {global_step}: "
                                   f"Source loss={validation_results['source_results']['val_loss']:.4f}")
                
                # End epoch with summary
                epoch_metrics = {
                    'epoch_accuracy': 0.6 + epoch * 0.1,
                    'epoch_map': 0.5 + epoch * 0.08
                }
                monitor.end_epoch(epoch_metrics)
        
        # Verify monitoring results
        final_summary = monitor.get_monitoring_summary()
        
        assert final_summary['step_count'] == 119  # 3 epochs * 40 steps - 1 (0-indexed)
        assert final_summary['epoch_count'] == 2   # Last epoch processed
        
        if monitor.ema_handler:
            ema_stats = monitor.ema_handler.get_ema_statistics()
            assert ema_stats['update_count'] > 0
        
        if monitor.loss_tracker:
            loss_summary = monitor.loss_tracker.get_loss_summary()
            assert loss_summary['step_count'] == 119
        
        # Check that files were created
        save_dir = Path(temp_dir)
        summary_file = save_dir / "training_summary.json"
        assert summary_file.exists(), "Training summary should be saved"
        
        # Check metrics logging
        metrics_dir = save_dir / "metrics"
        if metrics_dir.exists():
            log_files = list(metrics_dir.rglob("*.jsonl")) + list(metrics_dir.rglob("*.csv"))
            logger.info(f"    Created {len(log_files)} metrics log files")
    
    logger.info("✅ Training Monitor Integration tests passed")


def test_optimizer_scheduler_integration():
    """Test integration with optimizers and schedulers from previous waves."""
    logger.info("Testing Optimizer & Scheduler Integration...")
    
    model = MockRFDETRModel()
    
    # Create optimizer from Wave 1
    optimizer, gradient_handler = create_rfdetr_optimizer(model, "surveillance_optimized")
    
    # Create scheduler from Wave 2
    scheduler = create_cosine_scheduler_with_warmup(
        optimizer, 
        total_epochs=10, 
        warmup_epochs=2,
        min_lr_ratio=0.01
    )
    
    # Create monitoring from Wave 3
    monitor = create_training_monitor(
        model,
        save_dir=tempfile.mkdtemp(),
        validation_frequency=10
    )
    
    # Simulate integrated training workflow
    with monitor:
        for epoch in range(5):
            monitor.start_epoch(epoch)
            scheduler.step(epoch)
            
            for step in range(20):
                global_step = epoch * 20 + step
                
                # Simulate forward pass
                batch_size = 4
                inputs = torch.randn(batch_size, 3, 224, 224)
                
                # Mock loss computation
                outputs = model(inputs)
                loss_dict = {
                    'loss': torch.tensor(1.5 - global_step * 0.01),
                    'loss_ce': torch.tensor(0.9 - global_step * 0.006),
                    'loss_bbox': torch.tensor(0.4 - global_step * 0.003),
                    'loss_giou': torch.tensor(0.2 - global_step * 0.001)
                }
                
                # Simulate backward pass
                total_loss = loss_dict['loss']
                
                # Mock gradients (in real training, this would be total_loss.backward())
                for param in model.parameters():
                    if param.grad is None:
                        param.grad = torch.randn_like(param.data) * 0.01
                
                # Apply gradient clipping
                gradient_handler.clip_gradients()
                
                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()
                
                # Monitor step
                additional_metrics = {
                    'learning_rate': scheduler.get_last_lr()[0],
                    'gradient_norm': gradient_handler.get_gradient_norm(),
                    'optimizer_step': optimizer.get_step_count() if hasattr(optimizer, 'get_step_count') else global_step
                }
                
                monitor.step(loss_dict, global_step, epoch, additional_metrics)
            
            monitor.end_epoch({'epoch_lr': scheduler.get_last_lr()[0]})
    
    # Verify integration worked
    final_summary = monitor.get_monitoring_summary()
    assert final_summary['step_count'] == 99  # 5 epochs * 20 steps - 1
    
    logger.info("✅ Optimizer & Scheduler Integration tests passed")


def run_comprehensive_test():
    """Run all comprehensive tests."""
    logger.info("🚀 Starting Comprehensive RF-DETR Training Monitoring Tests")
    logger.info("=" * 80)
    
    try:
        # Test individual components
        test_ema_handler()
        test_loss_tracker()
        test_metrics_logger()
        
        # Test comprehensive integration
        test_training_monitor_integration()
        
        # Test cross-wave integration
        test_optimizer_scheduler_integration()
        
        logger.info("=" * 80)
        logger.info("🎉 ALL TESTS PASSED - Wave 3 Training Monitoring Infrastructure Complete!")
        logger.info("")
        logger.info("Wave 3 Components Successfully Implemented:")
        logger.info("✅ EMA Handler - Model stabilization with dynamic decay scheduling")
        logger.info("✅ Loss Tracker - Comprehensive loss analysis with anomaly detection")
        logger.info("✅ Metrics Logger - Multi-format logging with async writing")
        logger.info("✅ Training Monitor - Orchestrated monitoring with validation integration")
        logger.info("✅ Cross-Wave Integration - Full compatibility with Waves 1 & 2")
        logger.info("")
        logger.info("Ready to proceed to Wave 4: MTMMC Scene-Specific Optimization")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)