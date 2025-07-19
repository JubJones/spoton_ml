"""
Automated Report Generation System for Phase 1.3.2.

This module implements comprehensive automated report generation including:
1. HTML analysis reports with interactive visualizations
2. Executive summary dashboards
3. Trend analysis charts
4. Performance degradation alert system
5. Multi-format report exports (HTML, PDF, JSON)
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
import cv2
import torch
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from jinja2 import Template, Environment, FileSystemLoader
import base64
from io import BytesIO

logger = logging.getLogger(__name__)


@dataclass
class ReportMetadata:
    """Metadata for generated reports."""
    report_id: str
    timestamp: datetime
    report_type: str
    model_names: List[str]
    dataset_info: Dict[str, Any]
    analysis_duration: float
    total_frames_analyzed: int
    

@dataclass
class AlertRule:
    """Performance alert rule definition."""
    metric_name: str
    threshold: float
    comparison: str  # 'less_than', 'greater_than', 'equals'
    severity: str    # 'low', 'medium', 'high', 'critical'
    description: str
    

@dataclass
class PerformanceAlert:
    """Performance degradation alert."""
    alert_id: str
    timestamp: datetime
    model_name: str
    metric_name: str
    current_value: float
    threshold: float
    severity: str
    description: str
    suggested_actions: List[str]


class TrendAnalyzer:
    """Analyzes performance trends over time."""
    
    def __init__(self):
        self.historical_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
    def add_metrics_snapshot(self, model_name: str, metrics: Dict[str, Any], timestamp: datetime):
        """Add a metrics snapshot for trend analysis."""
        snapshot = {
            'timestamp': timestamp,
            'metrics': metrics.copy()
        }
        self.historical_data[model_name].append(snapshot)
        
    def analyze_trends(self, model_name: str, lookback_days: int = 30) -> Dict[str, Any]:
        """Analyze performance trends for a model."""
        if model_name not in self.historical_data:
            return {}
        
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        recent_data = [
            snapshot for snapshot in self.historical_data[model_name]
            if snapshot['timestamp'] >= cutoff_date
        ]
        
        if len(recent_data) < 2:
            return {'error': 'Insufficient historical data for trend analysis'}
        
        trends = {}
        
        # Analyze key metrics trends
        key_metrics = ['precision', 'recall', 'f1_score', 'average_processing_time']
        
        for metric in key_metrics:
            values = []
            timestamps = []
            
            for snapshot in recent_data:
                if metric in snapshot['metrics']:
                    values.append(snapshot['metrics'][metric])
                    timestamps.append(snapshot['timestamp'])
            
            if len(values) >= 2:
                trend_analysis = self._calculate_trend(values, timestamps)
                trends[metric] = trend_analysis
        
        return trends
    
    def _calculate_trend(self, values: List[float], timestamps: List[datetime]) -> Dict[str, Any]:
        """Calculate trend statistics for a metric."""
        if len(values) < 2:
            return {}
        
        # Convert timestamps to numeric values (days since first timestamp)
        first_timestamp = timestamps[0]
        x = [(ts - first_timestamp).total_seconds() / 86400 for ts in timestamps]
        y = values
        
        # Calculate linear trend
        slope, intercept = np.polyfit(x, y, 1)
        
        # Calculate trend statistics
        latest_value = values[-1]
        earliest_value = values[0]
        absolute_change = latest_value - earliest_value
        percent_change = (absolute_change / earliest_value * 100) if earliest_value != 0 else 0
        
        # Determine trend direction
        if abs(percent_change) < 1:  # Less than 1% change
            trend_direction = 'stable'
        elif percent_change > 0:
            trend_direction = 'improving' if slope > 0 else 'stable'
        else:
            trend_direction = 'degrading' if slope < 0 else 'stable'
        
        return {
            'slope': slope,
            'latest_value': latest_value,
            'earliest_value': earliest_value,
            'absolute_change': absolute_change,
            'percent_change': percent_change,
            'trend_direction': trend_direction,
            'data_points': len(values)
        }


class AlertManager:
    """Manages performance degradation alerts."""
    
    def __init__(self):
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: List[PerformanceAlert] = []
        self._setup_default_rules()
        
    def _setup_default_rules(self):
        """Setup default alert rules."""
        default_rules = [
            AlertRule(
                metric_name='f1_score',
                threshold=0.7,
                comparison='less_than',
                severity='high',
                description='F1 score below acceptable threshold'
            ),
            AlertRule(
                metric_name='precision',
                threshold=0.75,
                comparison='less_than',
                severity='medium',
                description='Precision below target threshold'
            ),
            AlertRule(
                metric_name='recall',
                threshold=0.75,
                comparison='less_than',
                severity='medium',
                description='Recall below target threshold'
            ),
            AlertRule(
                metric_name='average_processing_time',
                threshold=0.1,  # 100ms
                comparison='greater_than',
                severity='medium',
                description='Processing time exceeds real-time requirements'
            ),
            AlertRule(
                metric_name='success_rate',
                threshold=0.95,
                comparison='less_than',
                severity='high',
                description='Model reliability below acceptable threshold'
            )
        ]
        
        self.alert_rules.extend(default_rules)
    
    def add_alert_rule(self, rule: AlertRule):
        """Add custom alert rule."""
        self.alert_rules.append(rule)
    
    def check_alerts(self, model_name: str, metrics: Dict[str, Any]) -> List[PerformanceAlert]:
        """Check metrics against alert rules and generate alerts."""
        new_alerts = []
        
        for rule in self.alert_rules:
            if rule.metric_name not in metrics:
                continue
            
            current_value = metrics[rule.metric_name]
            is_triggered = False
            
            if rule.comparison == 'less_than' and current_value < rule.threshold:
                is_triggered = True
            elif rule.comparison == 'greater_than' and current_value > rule.threshold:
                is_triggered = True
            elif rule.comparison == 'equals' and abs(current_value - rule.threshold) < 1e-6:
                is_triggered = True
            
            if is_triggered:
                alert = PerformanceAlert(
                    alert_id=f"{model_name}_{rule.metric_name}_{datetime.now().isoformat()}",
                    timestamp=datetime.now(),
                    model_name=model_name,
                    metric_name=rule.metric_name,
                    current_value=current_value,
                    threshold=rule.threshold,
                    severity=rule.severity,
                    description=rule.description,
                    suggested_actions=self._generate_suggested_actions(rule, current_value)
                )
                new_alerts.append(alert)
                self.active_alerts.append(alert)
        
        return new_alerts
    
    def _generate_suggested_actions(self, rule: AlertRule, current_value: float) -> List[str]:
        """Generate suggested actions based on alert type."""
        actions = []
        
        if rule.metric_name == 'f1_score':
            actions.extend([
                "Review training data quality and balance",
                "Consider hyperparameter tuning",
                "Analyze failure cases for common patterns",
                "Evaluate ensemble methods"
            ])
        elif rule.metric_name == 'precision':
            actions.extend([
                "Increase confidence threshold to reduce false positives",
                "Review and improve training data annotation quality",
                "Consider additional negative examples in training"
            ])
        elif rule.metric_name == 'recall':
            actions.extend([
                "Decrease confidence threshold to catch more detections",
                "Increase training data diversity",
                "Review hard negative mining strategies"
            ])
        elif rule.metric_name == 'average_processing_time':
            actions.extend([
                "Profile model inference for bottlenecks",
                "Consider model quantization or pruning",
                "Optimize preprocessing pipeline",
                "Evaluate hardware acceleration options"
            ])
        elif rule.metric_name == 'success_rate':
            actions.extend([
                "Review model initialization and loading procedures",
                "Check for memory or hardware resource constraints",
                "Validate input data format and preprocessing"
            ])
        
        return actions
    
    def get_alerts_by_severity(self, severity: str) -> List[PerformanceAlert]:
        """Get active alerts by severity level."""
        return [alert for alert in self.active_alerts if alert.severity == severity]
    
    def clear_old_alerts(self, hours: int = 24):
        """Clear alerts older than specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        self.active_alerts = [
            alert for alert in self.active_alerts 
            if alert.timestamp >= cutoff_time
        ]


class HTMLReportGenerator:
    """Generates comprehensive HTML reports."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup Jinja2 environment
        self.template_dir = output_dir / 'templates'
        self.template_dir.mkdir(exist_ok=True)
        self._create_html_templates()
        
        self.env = Environment(loader=FileSystemLoader(str(self.template_dir)))
        
    def _create_html_templates(self):
        """Create HTML templates for reports."""
        
        # Main report template
        main_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ report_title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { border-bottom: 3px solid #007bff; margin-bottom: 30px; padding-bottom: 20px; }
        .title { color: #007bff; margin: 0; font-size: 2.5em; }
        .subtitle { color: #666; margin: 10px 0 0 0; font-size: 1.2em; }
        .metadata { background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .section { margin: 30px 0; }
        .section-title { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; margin-bottom: 20px; }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }
        .metric-card { background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; border-left: 4px solid #007bff; }
        .metric-value { font-size: 2em; font-weight: bold; color: #007bff; }
        .metric-label { color: #666; margin-top: 10px; }
        .alert { padding: 15px; margin: 10px 0; border-radius: 5px; }
        .alert-critical { background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
        .alert-high { background: #fff3cd; border: 1px solid #ffeaa7; color: #856404; }
        .alert-medium { background: #cce5ff; border: 1px solid #b8daff; color: #004085; }
        .image-container { text-align: center; margin: 20px 0; }
        .image-container img { max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .trend-indicator { display: inline-block; padding: 5px 10px; border-radius: 20px; font-size: 0.9em; font-weight: bold; }
        .trend-improving { background: #d4edda; color: #155724; }
        .trend-stable { background: #e2e3e5; color: #383d41; }
        .trend-degrading { background: #f8d7da; color: #721c24; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 class="title">{{ report_title }}</h1>
            <p class="subtitle">{{ report_subtitle }}</p>
        </div>
        
        <div class="metadata">
            <strong>Report Generated:</strong> {{ timestamp }}<br>
            <strong>Analysis Duration:</strong> {{ analysis_duration }}<br>
            <strong>Models Analyzed:</strong> {{ models_analyzed }}<br>
            <strong>Total Frames:</strong> {{ total_frames }}
        </div>
        
        {% if alerts %}
        <div class="section">
            <h2 class="section-title">🚨 Active Alerts</h2>
            {% for alert in alerts %}
            <div class="alert alert-{{ alert.severity }}">
                <strong>{{ alert.model_name }}</strong> - {{ alert.description }}<br>
                <small>{{ alert.metric_name }}: {{ "%.3f"|format(alert.current_value) }} (threshold: {{ "%.3f"|format(alert.threshold) }})</small>
            </div>
            {% endfor %}
        </div>
        {% endif %}
        
        <div class="section">
            <h2 class="section-title">📊 Performance Overview</h2>
            <div class="metric-grid">
                {% for metric in key_metrics %}
                <div class="metric-card">
                    <div class="metric-value">{{ "%.3f"|format(metric.value) }}</div>
                    <div class="metric-label">{{ metric.label }}</div>
                    {% if metric.trend %}
                    <div class="trend-indicator trend-{{ metric.trend.direction }}">
                        {{ metric.trend.direction|title }} ({{ "%.1f"|format(metric.trend.percent_change) }}%)
                    </div>
                    {% endif %}
                </div>
                {% endfor %}
            </div>
        </div>
        
        {% if visualizations %}
        <div class="section">
            <h2 class="section-title">📈 Visualizations</h2>
            {% for viz in visualizations %}
            <div class="image-container">
                <h3>{{ viz.title }}</h3>
                <img src="data:image/png;base64,{{ viz.image_base64 }}" alt="{{ viz.title }}">
                {% if viz.description %}
                <p>{{ viz.description }}</p>
                {% endif %}
            </div>
            {% endfor %}
        </div>
        {% endif %}
        
        {% if model_comparison %}
        <div class="section">
            <h2 class="section-title">🔄 Model Comparison</h2>
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1-Score</th>
                        <th>Processing Time (ms)</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {% for model in model_comparison %}
                    <tr>
                        <td><strong>{{ model.name }}</strong></td>
                        <td>{{ "%.3f"|format(model.precision) }}</td>
                        <td>{{ "%.3f"|format(model.recall) }}</td>
                        <td>{{ "%.3f"|format(model.f1_score) }}</td>
                        <td>{{ "%.1f"|format(model.processing_time * 1000) }}</td>
                        <td>
                            {% if model.status == 'good' %}
                            <span style="color: green;">✓ Good</span>
                            {% elif model.status == 'warning' %}
                            <span style="color: orange;">⚠ Warning</span>
                            {% else %}
                            <span style="color: red;">✗ Issues</span>
                            {% endif %}
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}
        
        {% if recommendations %}
        <div class="section">
            <h2 class="section-title">💡 Recommendations</h2>
            <ul>
                {% for rec in recommendations %}
                <li>{{ rec }}</li>
                {% endfor %}
            </ul>
        </div>
        {% endif %}
        
        <div class="section">
            <h2 class="section-title">📋 Technical Details</h2>
            <p><strong>Report ID:</strong> {{ report_id }}</p>
            <p><strong>Analysis Framework:</strong> Enhanced Detection Analysis - Phase 1</p>
            <p><strong>Generated by:</strong> Automated Reporting System</p>
        </div>
    </div>
</body>
</html>
        '''
        
        template_path = self.template_dir / 'main_report.html'
        with open(template_path, 'w') as f:
            f.write(main_template)
    
    def _encode_image_to_base64(self, image_path: Path) -> str:
        """Convert image file to base64 string."""
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
            return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            return ""
    
    def generate_html_report(self, report_data: Dict[str, Any], 
                           visualization_paths: List[Path] = None) -> Path:
        """Generate comprehensive HTML report."""
        template = self.env.get_template('main_report.html')
        
        # Prepare visualization data
        visualizations = []
        if visualization_paths:
            for viz_path in visualization_paths:
                if viz_path.exists():
                    viz_data = {
                        'title': viz_path.stem.replace('_', ' ').title(),
                        'image_base64': self._encode_image_to_base64(viz_path),
                        'description': f"Analysis visualization: {viz_path.stem}"
                    }
                    visualizations.append(viz_data)
        
        # Prepare template data
        template_data = {
            'report_title': report_data.get('title', 'Detection Analysis Report'),
            'report_subtitle': report_data.get('subtitle', 'Automated Performance Analysis'),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'analysis_duration': report_data.get('analysis_duration', 'N/A'),
            'models_analyzed': ', '.join(report_data.get('models', [])),
            'total_frames': report_data.get('total_frames', 0),
            'report_id': report_data.get('report_id', 'Unknown'),
            'alerts': report_data.get('alerts', []),
            'key_metrics': report_data.get('key_metrics', []),
            'visualizations': visualizations,
            'model_comparison': report_data.get('model_comparison', []),
            'recommendations': report_data.get('recommendations', [])
        }
        
        # Render template
        html_content = template.render(**template_data)
        
        # Save HTML report
        report_path = self.output_dir / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return report_path


class DashboardGenerator:
    """Generates executive summary dashboards."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_executive_dashboard(self, analysis_results: Dict[str, Any]) -> Dict[str, Path]:
        """Generate executive dashboard with key insights."""
        dashboard_files = {}
        
        # 1. Performance summary chart
        summary_chart = self._create_performance_summary_chart(analysis_results)
        if summary_chart:
            dashboard_files['performance_summary'] = summary_chart
        
        # 2. Alert status dashboard
        alert_dashboard = self._create_alert_status_dashboard(analysis_results)
        if alert_dashboard:
            dashboard_files['alert_status'] = alert_dashboard
        
        # 3. Trend analysis dashboard
        trend_dashboard = self._create_trend_dashboard(analysis_results)
        if trend_dashboard:
            dashboard_files['trend_analysis'] = trend_dashboard
        
        return dashboard_files
    
    def _create_performance_summary_chart(self, analysis_results: Dict[str, Any]) -> Optional[Path]:
        """Create high-level performance summary chart."""
        try:
            models = analysis_results.get('models', {})
            if not models:
                return None
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Executive Performance Dashboard', fontsize=20, fontweight='bold')
            
            model_names = list(models.keys())
            
            # 1. Overall Performance (F1 Scores)
            f1_scores = [models[model].get('f1_score', 0) for model in model_names]
            bars1 = axes[0, 0].bar(model_names, f1_scores, color='skyblue')
            axes[0, 0].set_title('Model Performance (F1-Score)', fontsize=14, fontweight='bold')
            axes[0, 0].set_ylabel('F1-Score')
            axes[0, 0].set_ylim(0, 1)
            
            # Add value labels
            for bar, score in zip(bars1, f1_scores):
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # 2. Processing Speed
            proc_times = [models[model].get('average_processing_time', 0) * 1000 for model in model_names]
            bars2 = axes[0, 1].bar(model_names, proc_times, color='lightcoral')
            axes[0, 1].set_title('Processing Speed (ms per frame)', fontsize=14, fontweight='bold')
            axes[0, 1].set_ylabel('Processing Time (ms)')
            
            for bar, time in zip(bars2, proc_times):
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                               f'{time:.1f}', ha='center', va='bottom', fontweight='bold')
            
            # 3. Reliability (Success Rate)
            success_rates = [models[model].get('success_rate', 0) for model in model_names]
            bars3 = axes[1, 0].bar(model_names, success_rates, color='lightgreen')
            axes[1, 0].set_title('Model Reliability (Success Rate)', fontsize=14, fontweight='bold')
            axes[1, 0].set_ylabel('Success Rate')
            axes[1, 0].set_ylim(0, 1)
            
            for bar, rate in zip(bars3, success_rates):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{rate:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # 4. Overall Score (weighted combination)
            overall_scores = []
            for model in model_names:
                f1 = models[model].get('f1_score', 0)
                speed_score = max(0, 1 - models[model].get('average_processing_time', 0) / 0.5)  # Penalty for >500ms
                reliability = models[model].get('success_rate', 0)
                overall = (f1 * 0.5 + speed_score * 0.2 + reliability * 0.3)
                overall_scores.append(overall)
            
            bars4 = axes[1, 1].bar(model_names, overall_scores, color='gold')
            axes[1, 1].set_title('Overall Score (Weighted)', fontsize=14, fontweight='bold')
            axes[1, 1].set_ylabel('Overall Score')
            axes[1, 1].set_ylim(0, 1)
            
            for bar, score in zip(bars4, overall_scores):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Rotate x-axis labels if needed
            for ax in axes.flat:
                ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            chart_path = self.output_dir / 'executive_performance_dashboard.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"Error creating performance summary chart: {e}")
            return None
    
    def _create_alert_status_dashboard(self, analysis_results: Dict[str, Any]) -> Optional[Path]:
        """Create alert status dashboard."""
        try:
            alerts = analysis_results.get('alerts', [])
            if not alerts:
                # Create "no alerts" dashboard
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, '✅ No Active Alerts\nAll Systems Operating Normally', 
                       ha='center', va='center', fontsize=20, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                ax.set_title('Alert Status Dashboard', fontsize=16, fontweight='bold', pad=20)
                
                chart_path = self.output_dir / 'alert_status_dashboard.png'
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                return chart_path
            
            # Count alerts by severity
            severity_counts = defaultdict(int)
            for alert in alerts:
                severity_counts[alert.get('severity', 'unknown')] += 1
            
            # Create pie chart
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle('Alert Status Dashboard', fontsize=16, fontweight='bold')
            
            # Pie chart of alert severities
            severities = list(severity_counts.keys())
            counts = list(severity_counts.values())
            colors = {'critical': 'red', 'high': 'orange', 'medium': 'yellow', 'low': 'lightblue'}
            pie_colors = [colors.get(sev, 'gray') for sev in severities]
            
            ax1.pie(counts, labels=severities, colors=pie_colors, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Alerts by Severity', fontweight='bold')
            
            # Recent alerts timeline (simplified)
            models_with_alerts = list(set(alert.get('model_name', 'unknown') for alert in alerts))
            model_alert_counts = [sum(1 for alert in alerts if alert.get('model_name') == model) 
                                for model in models_with_alerts]
            
            bars = ax2.bar(models_with_alerts, model_alert_counts, color='orange')
            ax2.set_title('Alerts by Model', fontweight='bold')
            ax2.set_ylabel('Number of Alerts')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, count in zip(bars, model_alert_counts):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        str(count), ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            chart_path = self.output_dir / 'alert_status_dashboard.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"Error creating alert status dashboard: {e}")
            return None
    
    def _create_trend_dashboard(self, analysis_results: Dict[str, Any]) -> Optional[Path]:
        """Create trend analysis dashboard."""
        try:
            trends = analysis_results.get('trends', {})
            if not trends:
                return None
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Performance Trend Dashboard', fontsize=16, fontweight='bold')
            
            # Extract trend data for key metrics
            models = list(trends.keys())
            
            metrics_to_plot = ['f1_score', 'precision', 'recall', 'average_processing_time']
            metric_titles = ['F1-Score Trends', 'Precision Trends', 'Recall Trends', 'Processing Time Trends']
            
            for idx, (metric, title) in enumerate(zip(metrics_to_plot, metric_titles)):
                ax = axes[idx // 2, idx % 2]
                
                for model in models:
                    model_trends = trends[model]
                    if metric in model_trends:
                        trend_data = model_trends[metric]
                        # Simplified trend visualization
                        direction = trend_data.get('trend_direction', 'stable')
                        percent_change = trend_data.get('percent_change', 0)
                        
                        # Create simple bar showing trend direction and magnitude
                        color = 'green' if direction == 'improving' else 'red' if direction == 'degrading' else 'gray'
                        ax.bar(model, percent_change, color=color, alpha=0.7)
                
                ax.set_title(title, fontweight='bold')
                ax.set_ylabel('Percent Change (%)')
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            chart_path = self.output_dir / 'trend_analysis_dashboard.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"Error creating trend dashboard: {e}")
            return None


class AutomatedReportingSystem:
    """Main automated reporting system integrating all components."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.trend_analyzer = TrendAnalyzer()
        self.alert_manager = AlertManager()
        self.html_generator = HTMLReportGenerator(output_dir / 'html')
        self.dashboard_generator = DashboardGenerator(output_dir / 'dashboards')
        
    def generate_comprehensive_report(self, analysis_results: Dict[str, Any],
                                    visualization_paths: List[Path] = None) -> Dict[str, Path]:
        """Generate comprehensive automated report."""
        logger.info("Generating comprehensive automated report...")
        
        generated_files = {}
        
        # Create report metadata
        report_metadata = ReportMetadata(
            report_id=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            report_type='Comprehensive Detection Analysis',
            model_names=list(analysis_results.get('models', {}).keys()),
            dataset_info=analysis_results.get('dataset_info', {}),
            analysis_duration=analysis_results.get('analysis_duration', 0.0),
            total_frames_analyzed=analysis_results.get('total_frames', 0)
        )
        
        # Check for alerts
        alerts = []
        for model_name, model_metrics in analysis_results.get('models', {}).items():
            model_alerts = self.alert_manager.check_alerts(model_name, model_metrics)
            alerts.extend([asdict(alert) for alert in model_alerts])
        
        # Generate trend analysis
        trends = {}
        for model_name, model_metrics in analysis_results.get('models', {}).items():
            self.trend_analyzer.add_metrics_snapshot(model_name, model_metrics, datetime.now())
            model_trends = self.trend_analyzer.analyze_trends(model_name)
            if model_trends:
                trends[model_name] = model_trends
        
        # Prepare report data
        report_data = {
            'title': 'Enhanced Detection Analysis Report',
            'subtitle': 'Phase 1: Multi-dimensional Failure Analysis',
            'report_id': report_metadata.report_id,
            'analysis_duration': f"{report_metadata.analysis_duration:.2f} seconds",
            'models': list(report_metadata.model_names),
            'total_frames': report_metadata.total_frames_analyzed,
            'alerts': alerts,
            'trends': trends,
            'key_metrics': self._extract_key_metrics(analysis_results),
            'model_comparison': self._prepare_model_comparison(analysis_results),
            'recommendations': self._generate_recommendations(analysis_results, alerts)
        }
        
        # Generate HTML report
        html_report = self.html_generator.generate_html_report(
            report_data, visualization_paths
        )
        generated_files['html_report'] = html_report
        
        # Generate executive dashboards
        dashboard_data = {**analysis_results, 'alerts': alerts, 'trends': trends}
        dashboards = self.dashboard_generator.generate_executive_dashboard(dashboard_data)
        generated_files.update(dashboards)
        
        # Generate JSON report
        json_report = self._generate_json_report(report_data, report_metadata)
        generated_files['json_report'] = json_report
        
        # Generate summary statistics
        summary_stats = self._generate_summary_statistics(analysis_results, alerts)
        generated_files['summary_statistics'] = summary_stats
        
        logger.info(f"Automated report generation complete. Generated {len(generated_files)} files.")
        return generated_files
    
    def _extract_key_metrics(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract key metrics for dashboard display."""
        models = analysis_results.get('models', {})
        if not models:
            return []
        
        # Calculate overall averages
        total_f1 = sum(model.get('f1_score', 0) for model in models.values())
        total_precision = sum(model.get('precision', 0) for model in models.values())
        total_recall = sum(model.get('recall', 0) for model in models.values())
        total_speed = sum(model.get('average_processing_time', 0) for model in models.values())
        
        num_models = len(models)
        
        key_metrics = [
            {
                'label': 'Average F1-Score',
                'value': total_f1 / num_models if num_models > 0 else 0,
                'trend': None  # Would need historical data
            },
            {
                'label': 'Average Precision',
                'value': total_precision / num_models if num_models > 0 else 0,
                'trend': None
            },
            {
                'label': 'Average Recall',
                'value': total_recall / num_models if num_models > 0 else 0,
                'trend': None
            },
            {
                'label': 'Average Processing Time (ms)',
                'value': (total_speed / num_models * 1000) if num_models > 0 else 0,
                'trend': None
            },
            {
                'label': 'Models Analyzed',
                'value': num_models,
                'trend': None
            },
            {
                'label': 'Total Frames Processed',
                'value': analysis_results.get('total_frames', 0),
                'trend': None
            }
        ]
        
        return key_metrics
    
    def _prepare_model_comparison(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare model comparison data for report."""
        models = analysis_results.get('models', {})
        comparison_data = []
        
        for model_name, metrics in models.items():
            f1_score = metrics.get('f1_score', 0)
            processing_time = metrics.get('average_processing_time', 0)
            success_rate = metrics.get('success_rate', 1)
            
            # Determine status
            status = 'good'
            if f1_score < 0.7 or processing_time > 0.1 or success_rate < 0.95:
                status = 'warning'
            if f1_score < 0.5 or processing_time > 0.2 or success_rate < 0.9:
                status = 'error'
            
            comparison_data.append({
                'name': model_name,
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1_score': f1_score,
                'processing_time': processing_time,
                'status': status
            })
        
        return comparison_data
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any], 
                                alerts: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Alert-based recommendations
        if alerts:
            critical_alerts = [a for a in alerts if a.get('severity') == 'critical']
            if critical_alerts:
                recommendations.append("🚨 Address critical performance issues immediately")
            
            high_alerts = [a for a in alerts if a.get('severity') == 'high']
            if high_alerts:
                recommendations.append("⚠️ Review high-priority performance degradation")
        
        # Model performance recommendations
        models = analysis_results.get('models', {})
        if models:
            best_model = max(models.items(), key=lambda x: x[1].get('f1_score', 0))
            worst_model = min(models.items(), key=lambda x: x[1].get('f1_score', 0))
            
            if best_model[1].get('f1_score', 0) > worst_model[1].get('f1_score', 0) + 0.1:
                recommendations.append(f"Consider using {best_model[0]} model for better performance")
            
            slow_models = [name for name, metrics in models.items() 
                          if metrics.get('average_processing_time', 0) > 0.1]
            if slow_models:
                recommendations.append(f"Optimize processing speed for: {', '.join(slow_models)}")
        
        # Generic recommendations
        if not recommendations:
            recommendations.extend([
                "Continue monitoring model performance regularly",
                "Consider A/B testing with ensemble methods",
                "Evaluate impact of different lighting conditions on performance"
            ])
        
        return recommendations
    
    def _generate_json_report(self, report_data: Dict[str, Any], 
                            metadata: ReportMetadata) -> Path:
        """Generate detailed JSON report."""
        json_data = {
            'metadata': asdict(metadata),
            'report_data': report_data,
            'generation_timestamp': datetime.now().isoformat()
        }
        
        json_path = self.output_dir / f'detailed_report_{metadata.report_id}.json'
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        return json_path
    
    def _generate_summary_statistics(self, analysis_results: Dict[str, Any],
                                   alerts: List[Dict[str, Any]]) -> Path:
        """Generate summary statistics file."""
        models = analysis_results.get('models', {})
        
        summary = {
            'executive_summary': {
                'total_models_analyzed': len(models),
                'total_frames_processed': analysis_results.get('total_frames', 0),
                'active_alerts': len(alerts),
                'critical_alerts': len([a for a in alerts if a.get('severity') == 'critical']),
                'analysis_timestamp': datetime.now().isoformat()
            },
            'performance_summary': {
                'best_performing_model': max(models.items(), key=lambda x: x[1].get('f1_score', 0))[0] if models else None,
                'fastest_model': min(models.items(), key=lambda x: x[1].get('average_processing_time', float('inf')))[0] if models else None,
                'average_f1_score': np.mean([m.get('f1_score', 0) for m in models.values()]) if models else 0,
                'average_processing_time_ms': np.mean([m.get('average_processing_time', 0) * 1000 for m in models.values()]) if models else 0
            },
            'alert_summary': {
                'alerts_by_severity': {
                    severity: len([a for a in alerts if a.get('severity') == severity])
                    for severity in ['critical', 'high', 'medium', 'low']
                },
                'alerts_by_model': {
                    model: len([a for a in alerts if a.get('model_name') == model])
                    for model in set(a.get('model_name', 'unknown') for a in alerts)
                }
            }
        }
        
        summary_path = self.output_dir / 'executive_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return summary_path