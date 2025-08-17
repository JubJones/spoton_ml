"""
Training Progress Tracker with Advanced Analytics
Real-time progress tracking, trend analysis, and performance prediction
"""
import logging
import time
import json
import numpy as np
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logging.debug("Matplotlib/seaborn not available. Plotting functionality disabled.")
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from scipy import stats

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.debug("Scikit-learn not available. Using NumPy fallback for regression.")
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class ProgressMetrics:
    """Progress metrics for training analysis."""
    
    # Time-based metrics
    elapsed_time: float
    estimated_total_time: float
    remaining_time: float
    
    # Progress metrics
    current_step: int
    total_steps: int
    progress_percentage: float
    
    # Performance trends
    loss_trend: str  # "improving", "plateauing", "deteriorating"
    loss_smoothed: float
    loss_velocity: float  # Rate of change
    
    # Learning dynamics
    learning_efficiency: float  # Progress per unit time
    convergence_prediction: Optional[int]  # Predicted convergence step
    
    # Resource utilization
    avg_step_time: float
    throughput: float  # Steps per second
    resource_efficiency: float
    
    # Quality metrics
    validation_trend: str
    best_metric_step: int
    steps_since_improvement: int


@dataclass
class TrendAnalysis:
    """Trend analysis results."""
    
    metric_name: str
    trend_direction: str  # "up", "down", "stable"
    trend_strength: float  # 0-1, strength of trend
    correlation: float  # Correlation with step number
    slope: float  # Rate of change
    r_squared: float  # Quality of trend fit
    confidence_interval: Tuple[float, float]
    
    # Predictions
    predicted_values: Optional[List[float]] = None
    prediction_steps: Optional[List[int]] = None


class PerformancePredictor:
    """Performance prediction using statistical models."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.models = {}
        self.scalers = {}
        
    def fit_trend(self, x: List[float], y: List[float], metric_name: str) -> TrendAnalysis:
        """Fit trend to time series data."""
        
        if len(x) < 10:  # Need minimum data points
            return TrendAnalysis(
                metric_name=metric_name,
                trend_direction="unknown",
                trend_strength=0.0,
                correlation=0.0,
                slope=0.0,
                r_squared=0.0,
                confidence_interval=(0.0, 0.0)
            )
        
        x_array = np.array(x).reshape(-1, 1)
        y_array = np.array(y)
        
        # Fit linear regression
        if SKLEARN_AVAILABLE:
            model = LinearRegression()
            model.fit(x_array, y_array)
            y_pred = model.predict(x_array)
            r_squared = model.score(x_array, y_array)
            slope = model.coef_[0]
        else:
            # NumPy fallback
            coeffs = np.polyfit(x, y, 1)
            slope = coeffs[0]
            y_pred = np.polyval(coeffs, x)
            # Calculate R-squared manually
            ss_res = np.sum((y_array - y_pred) ** 2)
            ss_tot = np.sum((y_array - np.mean(y_array)) ** 2)
            r_squared = 1 - (ss_res / (ss_tot + 1e-8))
            
            # Create simple model object for predictions
            class SimpleModel:
                def __init__(self, coeffs):
                    self.coef_ = [coeffs[0]]
                    self.intercept_ = coeffs[1]
                def predict(self, x):
                    return np.polyval(coeffs, x.flatten())
            model = SimpleModel(coeffs)
        
        # Calculate statistics
        correlation, _ = stats.pearsonr(x, y)
        
        # Determine trend direction and strength
        if abs(slope) < 1e-6:
            trend_direction = "stable"
            trend_strength = 0.0
        elif slope > 0:
            trend_direction = "up"
            trend_strength = min(abs(correlation), 1.0)
        else:
            trend_direction = "down"  
            trend_strength = min(abs(correlation), 1.0)
        
        # Calculate confidence interval (simplified)
        residuals = y_array - y_pred
        std_error = np.std(residuals)
        confidence_interval = (
            y_pred[-1] - 1.96 * std_error,
            y_pred[-1] + 1.96 * std_error
        )
        
        # Store model for predictions
        self.models[metric_name] = model
        
        return TrendAnalysis(
            metric_name=metric_name,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            correlation=correlation,
            slope=slope,
            r_squared=r_squared,
            confidence_interval=confidence_interval
        )
    
    def predict_convergence(
        self, 
        steps: List[int], 
        losses: List[float], 
        threshold: float = 0.001
    ) -> Optional[int]:
        """Predict when loss will converge."""
        
        if len(losses) < 20:  # Need sufficient data
            return None
        
        try:
            # Smooth the loss curve
            window = min(10, len(losses) // 5)
            smoothed_losses = np.convolve(losses, np.ones(window)/window, mode='valid')
            smoothed_steps = steps[window-1:]
            
            if len(smoothed_losses) < 10:
                return None
            
            # Fit exponential decay model
            # loss = a * exp(-b * step) + c
            # Linearize: log(loss - c) = log(a) - b * step
            
            min_loss = min(smoothed_losses)
            if min_loss <= 0:
                min_loss = 1e-8
            
            # Estimate asymptote
            asymptote = min_loss * 0.9
            
            # Transform data
            transformed_losses = np.log(np.maximum(smoothed_losses - asymptote, 1e-8))
            
            # Fit linear model to transformed data
            if SKLEARN_AVAILABLE:
                model = LinearRegression()
                model.fit(np.array(smoothed_steps).reshape(-1, 1), transformed_losses)
                convergence_step = (target_log_loss - model.intercept_) / model.coef_[0]
            else:
                # NumPy fallback
                coeffs = np.polyfit(smoothed_steps, transformed_losses, 1)
                convergence_step = (target_log_loss - coeffs[1]) / coeffs[0]
            
            # Already calculated convergence_step above
            
            return int(max(convergence_step, steps[-1]))
            
        except Exception as e:
            logger.debug(f"Convergence prediction failed: {e}")
            return None
    
    def predict_future_performance(
        self,
        steps: List[int],
        values: List[float],
        future_steps: int,
        metric_name: str
    ) -> Tuple[List[int], List[float]]:
        """Predict future performance values."""
        
        if metric_name not in self.models or len(steps) < 10:
            return [], []
        
        try:
            model = self.models[metric_name]
            
            # Generate future step numbers
            last_step = steps[-1]
            future_step_numbers = list(range(last_step + 1, last_step + future_steps + 1))
            
            # Predict future values
            future_x = np.array(future_step_numbers).reshape(-1, 1)
            future_values = model.predict(future_x)
            
            return future_step_numbers, future_values.tolist()
            
        except Exception as e:
            logger.debug(f"Future prediction failed: {e}")
            return [], []


class ProgressTracker:
    """Advanced training progress tracker with analytics."""
    
    def __init__(
        self,
        total_steps: int,
        update_frequency: int = 10,
        analysis_window: int = 100,
        save_dir: Optional[Path] = None
    ):
        self.total_steps = total_steps
        self.update_frequency = update_frequency
        self.analysis_window = analysis_window
        self.save_dir = save_dir or Path("progress_tracking")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.start_time = None
        self.last_update_time = None
        self.current_step = 0
        
        # Metrics storage
        self.step_history = deque(maxlen=1000)
        self.loss_history = deque(maxlen=1000)
        self.time_history = deque(maxlen=1000)
        self.validation_history = []
        
        # Performance tracking
        self.step_times = deque(maxlen=100)
        self.best_metrics = {}
        self.plateau_detection = defaultdict(int)
        
        # Analytics
        self.predictor = PerformancePredictor()
        self.trend_analyses = {}
        
        # Progress state
        self.is_started = False
        self.progress_data = {}
        
        logger.info(f"Progress Tracker initialized for {total_steps} steps")
    
    def start(self):
        """Start progress tracking."""
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.is_started = True
        logger.info("Progress tracking started")
    
    def update(
        self,
        step: int,
        loss: float,
        validation_metrics: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> ProgressMetrics:
        """Update progress with new training step."""
        
        if not self.is_started:
            self.start()
        
        current_time = time.time()
        self.current_step = step
        
        # Store metrics
        self.step_history.append(step)
        self.loss_history.append(loss)
        self.time_history.append(current_time)
        
        # Calculate step time
        if self.last_update_time:
            step_time = current_time - self.last_update_time
            self.step_times.append(step_time)
        
        self.last_update_time = current_time
        
        # Store validation metrics
        if validation_metrics:
            validation_entry = {
                'step': step,
                'timestamp': current_time,
                'metrics': validation_metrics.copy()
            }
            self.validation_history.append(validation_entry)
            
            # Update best metrics
            for metric, value in validation_metrics.items():
                if metric not in self.best_metrics or value > self.best_metrics[metric]:
                    self.best_metrics[metric] = value
        
        # Calculate progress metrics
        progress_metrics = self._calculate_progress_metrics()
        
        # Update trend analyses
        if step % self.update_frequency == 0:
            self._update_trend_analyses()
        
        # Save progress data
        self.progress_data = {
            'current_metrics': progress_metrics,
            'trend_analyses': self.trend_analyses,
            'best_metrics': self.best_metrics,
            'timestamp': current_time
        }
        
        return progress_metrics
    
    def _calculate_progress_metrics(self) -> ProgressMetrics:
        """Calculate comprehensive progress metrics."""
        
        if not self.step_history or not self.time_history:
            return ProgressMetrics(
                elapsed_time=0, estimated_total_time=0, remaining_time=0,
                current_step=0, total_steps=self.total_steps, progress_percentage=0,
                loss_trend="unknown", loss_smoothed=0, loss_velocity=0,
                learning_efficiency=0, convergence_prediction=None,
                avg_step_time=0, throughput=0, resource_efficiency=0,
                validation_trend="unknown", best_metric_step=0, steps_since_improvement=0
            )
        
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Progress calculations
        progress_percentage = (self.current_step / self.total_steps) * 100
        
        # Time estimates
        if len(self.step_times) > 0:
            avg_step_time = np.mean(self.step_times)
            remaining_steps = self.total_steps - self.current_step
            remaining_time = remaining_steps * avg_step_time
            estimated_total_time = elapsed_time + remaining_time
            throughput = 1.0 / avg_step_time if avg_step_time > 0 else 0
        else:
            avg_step_time = 0
            remaining_time = 0
            estimated_total_time = elapsed_time
            throughput = 0
        
        # Loss analysis
        loss_trend, loss_smoothed, loss_velocity = self._analyze_loss_trend()
        
        # Learning efficiency (progress per unit time)
        learning_efficiency = self.current_step / elapsed_time if elapsed_time > 0 else 0
        
        # Convergence prediction
        if len(self.loss_history) > 20:
            steps = list(self.step_history)[-len(self.loss_history):]
            losses = list(self.loss_history)
            convergence_prediction = self.predictor.predict_convergence(steps, losses)
        else:
            convergence_prediction = None
        
        # Resource efficiency (steps completed vs time taken)
        expected_time = self.current_step * 0.1  # Assume 0.1s per step baseline
        resource_efficiency = expected_time / elapsed_time if elapsed_time > 0 else 1.0
        
        # Validation analysis
        validation_trend = "unknown"
        best_metric_step = 0
        steps_since_improvement = 0
        
        if self.validation_history and self.best_metrics:
            # Find step where best metric was achieved
            primary_metric = list(self.best_metrics.keys())[0]
            best_value = self.best_metrics[primary_metric]
            
            for i, val_entry in enumerate(reversed(self.validation_history)):
                if val_entry['metrics'].get(primary_metric, 0) >= best_value:
                    best_metric_step = val_entry['step']
                    steps_since_improvement = self.current_step - best_metric_step
                    break
            
            # Determine validation trend
            if len(self.validation_history) >= 3:
                recent_values = [v['metrics'].get(primary_metric, 0) 
                               for v in self.validation_history[-3:]]
                if len(set(recent_values)) > 1:  # Not all same values
                    if recent_values[-1] > recent_values[0]:
                        validation_trend = "improving"
                    elif recent_values[-1] < recent_values[0]:
                        validation_trend = "deteriorating"
                    else:
                        validation_trend = "stable"
        
        return ProgressMetrics(
            elapsed_time=elapsed_time,
            estimated_total_time=estimated_total_time,
            remaining_time=remaining_time,
            current_step=self.current_step,
            total_steps=self.total_steps,
            progress_percentage=progress_percentage,
            loss_trend=loss_trend,
            loss_smoothed=loss_smoothed,
            loss_velocity=loss_velocity,
            learning_efficiency=learning_efficiency,
            convergence_prediction=convergence_prediction,
            avg_step_time=avg_step_time,
            throughput=throughput,
            resource_efficiency=resource_efficiency,
            validation_trend=validation_trend,
            best_metric_step=best_metric_step,
            steps_since_improvement=steps_since_improvement
        )
    
    def _analyze_loss_trend(self) -> Tuple[str, float, float]:
        """Analyze loss trend and calculate smoothed values."""
        
        if len(self.loss_history) < 5:
            return "unknown", 0.0, 0.0
        
        losses = list(self.loss_history)
        
        # Smooth loss using moving average
        window_size = min(10, len(losses) // 2)
        smoothed = np.convolve(losses, np.ones(window_size)/window_size, mode='same')
        loss_smoothed = float(smoothed[-1])
        
        # Calculate velocity (rate of change)
        if len(smoothed) >= 10:
            recent_window = smoothed[-10:]
            loss_velocity = (recent_window[-1] - recent_window[0]) / len(recent_window)
        else:
            loss_velocity = 0.0
        
        # Determine trend
        if abs(loss_velocity) < 1e-5:
            trend = "plateauing"
        elif loss_velocity < 0:
            trend = "improving"
        else:
            trend = "deteriorating"
        
        return trend, loss_smoothed, loss_velocity
    
    def _update_trend_analyses(self):
        """Update trend analyses for all metrics."""
        
        if len(self.step_history) < 10:
            return
        
        steps = list(self.step_history)
        
        # Analyze loss trend
        if len(self.loss_history) >= 10:
            losses = list(self.loss_history)
            self.trend_analyses['loss'] = self.predictor.fit_trend(
                steps[-len(losses):], losses, 'loss'
            )
        
        # Analyze validation metrics trends
        if len(self.validation_history) >= 5:
            for metric_name in self.best_metrics.keys():
                metric_values = []
                metric_steps = []
                
                for val_entry in self.validation_history:
                    if metric_name in val_entry['metrics']:
                        metric_values.append(val_entry['metrics'][metric_name])
                        metric_steps.append(val_entry['step'])
                
                if len(metric_values) >= 5:
                    self.trend_analyses[f'validation_{metric_name}'] = self.predictor.fit_trend(
                        metric_steps, metric_values, f'validation_{metric_name}'
                    )
    
    def generate_progress_report(self) -> Dict[str, Any]:
        """Generate comprehensive progress report."""
        
        if not self.progress_data:
            return {}
        
        current_metrics = self.progress_data['current_metrics']
        
        # Time formatting
        def format_time(seconds: float) -> str:
            if seconds < 60:
                return f"{seconds:.1f}s"
            elif seconds < 3600:
                return f"{seconds/60:.1f}m"
            else:
                return f"{seconds/3600:.1f}h"
        
        # Generate report
        report = {
            'summary': {
                'progress': f"{current_metrics.progress_percentage:.1f}%",
                'current_step': f"{current_metrics.current_step:,}/{self.total_steps:,}",
                'elapsed_time': format_time(current_metrics.elapsed_time),
                'remaining_time': format_time(current_metrics.remaining_time),
                'estimated_total': format_time(current_metrics.estimated_total_time)
            },
            'performance': {
                'throughput': f"{current_metrics.throughput:.2f} steps/sec",
                'avg_step_time': f"{current_metrics.avg_step_time*1000:.1f}ms",
                'resource_efficiency': f"{current_metrics.resource_efficiency:.2f}x",
                'learning_efficiency': f"{current_metrics.learning_efficiency:.2f} steps/sec"
            },
            'training_dynamics': {
                'loss_trend': current_metrics.loss_trend,
                'loss_smoothed': f"{current_metrics.loss_smoothed:.4f}",
                'loss_velocity': f"{current_metrics.loss_velocity:.2e}",
                'convergence_prediction': (
                    f"Step {current_metrics.convergence_prediction:,}" 
                    if current_metrics.convergence_prediction 
                    else "Unknown"
                )
            },
            'validation': {
                'trend': current_metrics.validation_trend,
                'best_metric_step': current_metrics.best_metric_step,
                'steps_since_improvement': current_metrics.steps_since_improvement,
                'best_metrics': self.best_metrics.copy()
            },
            'trend_analyses': {}
        }
        
        # Add trend analyses
        for metric_name, analysis in self.trend_analyses.items():
            report['trend_analyses'][metric_name] = {
                'direction': analysis.trend_direction,
                'strength': f"{analysis.trend_strength:.3f}",
                'correlation': f"{analysis.correlation:.3f}",
                'r_squared': f"{analysis.r_squared:.3f}",
                'slope': f"{analysis.slope:.2e}"
            }
        
        return report
    
    def save_progress(self, filename: Optional[str] = None):
        """Save progress data to file."""
        
        if filename is None:
            timestamp = int(time.time())
            filename = f"progress_{timestamp}.json"
        
        save_path = self.save_dir / filename
        
        # Prepare data for saving
        save_data = {
            'config': {
                'total_steps': self.total_steps,
                'analysis_window': self.analysis_window
            },
            'progress': self.progress_data,
            'history': {
                'steps': list(self.step_history),
                'losses': list(self.loss_history),
                'times': list(self.time_history),
                'validations': self.validation_history
            },
            'timestamp': time.time()
        }
        
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        logger.info(f"Progress saved to {save_path}")
    
    def generate_plots(self, save_plots: bool = True) -> Dict[str, Any]:
        """Generate progress visualization plots."""
        
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting not available - matplotlib/seaborn not installed")
            return {}
        
        if len(self.loss_history) < 10:
            logger.warning("Insufficient data for plotting")
            return {}
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress Analytics', fontsize=16)
        
        # Plot 1: Loss trend
        ax1 = axes[0, 0]
        steps = list(self.step_history)[-len(self.loss_history):]
        losses = list(self.loss_history)
        
        ax1.plot(steps, losses, alpha=0.7, label='Raw Loss')
        
        # Add smoothed loss if available
        if len(losses) >= 10:
            window = min(10, len(losses) // 5)
            smoothed = np.convolve(losses, np.ones(window)/window, mode='same')
            ax1.plot(steps, smoothed, linewidth=2, label='Smoothed Loss')
            
            # Add trend line
            if 'loss' in self.trend_analyses:
                analysis = self.trend_analyses['loss']
                future_steps, future_losses = self.predictor.predict_future_performance(
                    steps, losses, 50, 'loss'
                )
                if future_steps:
                    all_steps = steps + future_steps
                    model = self.predictor.models['loss']
                    trend_line = model.predict(np.array(all_steps).reshape(-1, 1))
                    ax1.plot(all_steps, trend_line, '--', alpha=0.8, label='Trend')
        
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Evolution')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Validation metrics
        ax2 = axes[0, 1]
        if self.validation_history and self.best_metrics:
            val_steps = [v['step'] for v in self.validation_history]
            
            for i, metric_name in enumerate(list(self.best_metrics.keys())[:3]):  # Max 3 metrics
                val_values = [v['metrics'].get(metric_name, 0) for v in self.validation_history]
                ax2.plot(val_steps, val_values, marker='o', label=metric_name)
            
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Validation Metric')
            ax2.set_title('Validation Progress')
            ax2.legend()
            ax2.grid(True)
        else:
            ax2.text(0.5, 0.5, 'No validation data', ha='center', va='center', 
                    transform=ax2.transAxes)
            ax2.set_title('Validation Progress')
        
        # Plot 3: Training speed
        ax3 = axes[1, 0]
        if len(self.step_times) >= 5:
            # Moving average of step times
            recent_times = list(self.step_times)
            time_steps = list(range(len(recent_times)))
            
            ax3.plot(time_steps, recent_times, alpha=0.7, label='Step Time')
            
            # Add moving average
            if len(recent_times) >= 10:
                window = min(10, len(recent_times) // 3)
                smoothed_times = np.convolve(recent_times, np.ones(window)/window, mode='same')
                ax3.plot(time_steps, smoothed_times, linewidth=2, label='Average')
            
            ax3.set_xlabel('Recent Steps')
            ax3.set_ylabel('Time (seconds)')
            ax3.set_title('Training Speed')
            ax3.legend()
            ax3.grid(True)
        else:
            ax3.text(0.5, 0.5, 'Insufficient timing data', ha='center', va='center',
                    transform=ax3.transAxes)
            ax3.set_title('Training Speed')
        
        # Plot 4: Progress summary
        ax4 = axes[1, 1]
        if hasattr(self, 'progress_data') and self.progress_data:
            metrics = self.progress_data['current_metrics']
            
            # Progress bar
            progress_bar = [metrics.progress_percentage / 100, 
                          1 - metrics.progress_percentage / 100]
            colors = ['#2ecc71', '#ecf0f1']
            
            wedges, texts, autotexts = ax4.pie(
                progress_bar, 
                labels=['Completed', 'Remaining'],
                colors=colors,
                autopct='%1.1f%%',
                startangle=90
            )
            
            ax4.set_title(f'Training Progress\n{metrics.current_step:,}/{self.total_steps:,} steps')
        else:
            ax4.text(0.5, 0.5, 'No progress data', ha='center', va='center',
                    transform=ax4.transAxes)
            ax4.set_title('Training Progress')
        
        plt.tight_layout()
        
        # Save plots if requested
        plot_info = {}
        if save_plots:
            timestamp = int(time.time())
            plot_path = self.save_dir / f"progress_plots_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plot_info['saved_path'] = str(plot_path)
            logger.info(f"Progress plots saved to {plot_path}")
        
        plot_info['figure'] = fig
        return plot_info
    
    def get_eta_string(self) -> str:
        """Get formatted ETA string."""
        if hasattr(self, 'progress_data') and self.progress_data:
            metrics = self.progress_data['current_metrics']
            remaining = metrics.remaining_time
            
            if remaining < 60:
                return f"{remaining:.0f}s"
            elif remaining < 3600:
                return f"{remaining/60:.0f}m {remaining%60:.0f}s"
            else:
                hours = int(remaining // 3600)
                minutes = int((remaining % 3600) // 60)
                return f"{hours}h {minutes}m"
        
        return "Unknown"
    
    def __str__(self) -> str:
        """String representation of current progress."""
        if hasattr(self, 'progress_data') and self.progress_data:
            metrics = self.progress_data['current_metrics']
            return (f"Progress: {metrics.progress_percentage:.1f}% "
                   f"({metrics.current_step:,}/{self.total_steps:,}) "
                   f"ETA: {self.get_eta_string()} "
                   f"Loss: {metrics.loss_smoothed:.4f} ({metrics.loss_trend})")
        
        return f"Progress: {self.current_step:,}/{self.total_steps:,}"


def create_progress_tracker(
    total_steps: int,
    update_frequency: int = 10,
    analysis_window: int = 100,
    save_dir: Optional[Path] = None
) -> ProgressTracker:
    """
    Create progress tracker with configuration.
    
    Args:
        total_steps: Total number of training steps
        update_frequency: Steps between trend analysis updates
        analysis_window: Window size for trend analysis
        save_dir: Directory for saving progress data
        
    Returns:
        Configured progress tracker
    """
    
    return ProgressTracker(
        total_steps=total_steps,
        update_frequency=update_frequency,
        analysis_window=analysis_window,
        save_dir=save_dir
    )


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing Training Progress Tracker")
    
    # Create tracker
    tracker = create_progress_tracker(
        total_steps=1000,
        update_frequency=5,
        save_dir=Path("test_progress")
    )
    
    print("✅ Progress tracker created")
    
    # Simulate training progress
    try:
        tracker.start()
        
        # Simulate training steps
        for step in range(0, 100, 5):
            # Simulate decreasing loss with some noise
            loss = 1.0 * np.exp(-step/50) + 0.1 * np.random.random()
            
            # Simulate validation every 25 steps
            validation_metrics = None
            if step % 25 == 0 and step > 0:
                map_score = 0.5 + 0.3 * (step / 100) + 0.1 * np.random.random()
                validation_metrics = {'mAP': map_score, 'precision': map_score * 1.1}
            
            # Update progress
            progress = tracker.update(step, loss, validation_metrics)
            
            if step % 20 == 0:  # Log every 20 steps
                print(f"  {tracker}")
            
            time.sleep(0.01)  # Simulate training time
        
        print("✅ Progress simulation completed")
        
        # Generate report
        report = tracker.generate_progress_report()
        print("📊 Progress Report:")
        print(f"  Progress: {report['summary']['progress']}")
        print(f"  ETA: {report['summary']['remaining_time']}")
        print(f"  Loss trend: {report['training_dynamics']['loss_trend']}")
        print(f"  Throughput: {report['performance']['throughput']}")
        
        # Test plotting (but don't show)
        if PLOTTING_AVAILABLE:
            plt.ioff()  # Turn off interactive mode
            plot_info = tracker.generate_plots(save_plots=False)
            if 'figure' in plot_info:
                plt.close(plot_info['figure'])
                print("✅ Plot generation successful")
        else:
            print("ℹ️  Plotting skipped (matplotlib not available)")
        
        # Save progress
        tracker.save_progress("test_progress.json")
        print("✅ Progress saved successfully")
        
    except Exception as e:
        print(f"❌ Progress tracking test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("✅ Training Progress Tracker testing completed")