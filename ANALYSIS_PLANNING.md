# ANALYSIS_PLANNING.md

## Executive Summary

This document outlines a focused per-scenario/camera detection analysis plan for the MTMMC vision pipeline using a single trained detection model. The goal is to identify detection performance across different environmental conditions and camera setups, collecting failure cases with 1 representative image per person ID that failed detection for detailed analysis.

## Current State Assessment

### Existing Capabilities
- ✅ Enhanced detection analysis pipeline (`src/pipelines/enhanced_detection_analysis.py`)
- ✅ Trained FasterRCNN model checkpoint: `checkpoints/7af7b38617994e41adbd761df223cf93/ckpt_best_eval_map_50.pth`
- ✅ MTMMC dataset loading with scene/camera management
- ✅ MLflow experiment tracking
- ✅ Failure case visualization and reporting

### Target Analysis Scope
- 🎯 **Single Model Analysis**: Focus on the existing trained FasterRCNN model
- 🎯 **Per-Scenario/Camera Analysis**: Evaluate performance across all scene/camera combinations
- 🎯 **Selective Failure Collection**: Save 1 representative image per person ID that failed detection
- 🎯 **Environmental Performance Mapping**: Identify which environments the model performs well/poorly

---

## PHASE 1: Per-Scenario/Camera Detection Analysis

### 1.1 Single Model Comprehensive Analysis
**Objective**: Evaluate trained FasterRCNN model performance across all scenarios/cameras

#### 1.1.1 Model Configuration
- ✅ **Pre-trained Model**: `checkpoints/7af7b38617994e41adbd761df223cf93/ckpt_best_eval_map_50.pth`
- ✅ **Model Type**: FasterRCNN (torchvision implementation)
- ✅ **Analysis Pipeline**: `src/pipelines/enhanced_detection_analysis.py`
- ✅ **Device**: CPU (FasterRCNN constraint)

#### 1.1.2 Scenario/Camera Coverage Analysis
- [ ] **Complete Coverage**: Run model on all scene/camera combinations in MTMMC dataset
- [ ] **Performance Mapping**: Generate performance metrics per scenario/camera pair
- [ ] **Environmental Correlation**: Identify which environments show good/poor performance
- [ ] **Failure Pattern Identification**: Collect failure cases with scene context

#### 1.1.3 Selective Failure Collection Strategy
- [ ] **Per-Person ID Sampling**: Save only 1 representative failure image per person ID
- [ ] **Failure Criteria**: Ground truth comparison with IoU threshold
- [ ] **Scene Context**: Include lighting conditions, crowd density, occlusion analysis
- [ ] **Image Organization**: Structure failure images by scene/camera/person_id

#### 1.1.4 Environmental Performance Analysis
- [ ] **Lighting Conditions**: Analyze performance across day/night/transition
- [ ] **Crowd Density**: Evaluate detection accuracy in low/medium/high density scenarios
- [ ] **Camera Perspectives**: Identify which camera angles/positions perform best/worst
- [ ] **Scene Complexity**: Correlate scene complexity with detection performance

### 1.2 Failure Case Documentation
**Objective**: Comprehensive documentation of detection failures

#### 1.2.1 Failure Image Collection
- [ ] **1-Per-Person Rule**: Collect maximum 1 failure image per person ID per scenario/camera
- [ ] **Representative Selection**: Choose most representative failure case if multiple exist
- [ ] **Dual Bounding Box Visualization**: 
  - **Ground Truth Boxes**: Display in RED color for missed detections
  - **Detected Boxes**: Display in BLUE color for all model predictions
  - **Color Coding**: Clear visual distinction between ground truth and predictions
- [ ] **Contextual Information**: Include scene conditions, frame metadata, failure reason

#### 1.2.2 Failure Analysis Framework
- [ ] **Failure Classification**: Categorize failures by type (occlusion, lighting, distance, etc.)
- [ ] **Statistical Analysis**: Generate failure rate statistics per scenario/camera
- [ ] **Performance Heatmaps**: Create visual performance maps across all combinations
- [ ] **Improvement Recommendations**: Identify specific areas for model enhancement

### 1.3 Results and Reporting
**Objective**: Comprehensive analysis documentation and insights

#### 1.3.1 Performance Summary Generation
- [ ] **Scenario Performance Matrix**: Performance metrics for each scene/camera combination
- [ ] **Best/Worst Environment Identification**: Highlight top and bottom performing environments
- [ ] **Failure Pattern Analysis**: Identify common failure patterns across scenarios
- [ ] **Environmental Correlation Analysis**: Link performance to environmental factors

#### 1.3.2 Actionable Insights
- [ ] **Environment-Specific Recommendations**: Suggest improvements for poor-performing scenarios
- [ ] **Training Data Insights**: Identify underrepresented scenarios in training
- [ ] **Model Limitation Analysis**: Document specific model weaknesses
- [ ] **Deployment Recommendations**: Suggest optimal deployment environments

---

## Implementation Details

### Configuration Requirements
- **Model Path**: `checkpoints/7af7b38617994e41adbd761df223cf93/ckpt_best_eval_map_50.pth`
- **Pipeline**: `src/pipelines/enhanced_detection_analysis.py`
- **Dataset**: MTMMC validation split with all scene/camera combinations
- **IoU Threshold**: 0.5 (configurable)
- **Confidence Threshold**: 0.5 (configurable)

### Output Structure
```
outputs/enhanced_detection_analysis/
├── failure_images/
│   ├── fasterrcnn_trained/
│   │   ├── failure_s01_c01_id1_frame123.png
│   │   ├── failure_s01_c02_id5_frame045.png
│   │   └── ...
├── reports/
│   ├── analysis_report.html
│   ├── model_comparison/
│   └── scenario_performance_matrix.csv
└── statistics/
    ├── failure_cases.csv
    ├── scenario_statistics.csv
    └── performance_plots.png
```

### Expected Outcomes
- **Performance Map**: Clear visualization of which scenarios/cameras perform well/poorly
- **Failure Gallery**: 1 representative failure image per person ID that failed detection
  - **Visual Format**: Ground truth boxes (RED) + detected boxes (BLUE) overlay
  - **Clear Distinction**: Easy identification of missed vs. detected persons
- **Statistical Analysis**: Quantitative breakdown of failure patterns
- **Actionable Insights**: Specific recommendations for model improvement and deployment

---

## Implementation Priority and Timeline

### Immediate Implementation (Phase 1)
1. **Model Configuration**: Set up FasterRCNN model with existing checkpoint
2. **Scenario Coverage**: Run analysis on all scene/camera combinations
3. **Failure Collection**: Implement 1-per-person-ID failure sampling
4. **Performance Mapping**: Generate scenario/camera performance matrix

### Success Metrics
- [ ] **Complete Coverage**: Analysis completed for all scene/camera combinations
- [ ] **Failure Documentation**: 1 representative failure image collected per person ID
- [ ] **Performance Insights**: Clear identification of best/worst performing environments
- [ ] **Actionable Recommendations**: Specific improvement suggestions for poor-performing scenarios

### Timeline
- **Setup and Configuration**: 1-2 days
- **Full Analysis Execution**: 3-5 days (depending on dataset size)
- **Report Generation**: 1-2 days
- **Total Duration**: 1 week

### Key Deliverables
1. **Failure Image Gallery**: Organized collection of representative failure cases
   - **Dual-Color Visualization**: RED (ground truth) + BLUE (detected) bounding boxes
   - **1-Per-Person Sampling**: Maximum 1 failure image per person ID per scenario/camera
2. **Performance Matrix**: Quantitative performance metrics per scenario/camera
3. **Environment Analysis Report**: Detailed breakdown of environmental factors
4. **Improvement Recommendations**: Specific actions for model enhancement

This focused analysis plan provides a clear, achievable path to understanding detection model performance across different environments, with actionable insights for improvement.