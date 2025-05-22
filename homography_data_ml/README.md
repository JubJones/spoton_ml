# Homography Data for ML Evaluation

This directory contains homography matrices and related data for evaluating tracking and Re-ID performance in the ML framework.

## Directory Structure

```
homography_data_ml/
├── factory/
│   ├── camera_1/
│   │   ├── homography_matrix.npy
│   │   └── exit_rules.yaml
│   ├── camera_2/
│   │   ├── homography_matrix.npy
│   │   └── exit_rules.yaml
│   └── ...
└── campus/
    ├── camera_1/
    │   ├── homography_matrix.npy
    │   └── exit_rules.yaml
    ├── camera_2/
    │   ├── homography_matrix.npy
    │   └── exit_rules.yaml
    └── ...
```

## File Formats

### Homography Matrices (.npy)

- NumPy array files containing 3x3 homography matrices
- Used for transforming coordinates between camera views
- Format: 3x3 numpy array of float32 values

### Exit Rules (exit_rules.yaml)

YAML files defining exit rules for each camera, with the following structure:

```yaml
exit_rules:
  - quadrant: "top_left"
    edge_threshold: 50.0
    target_cameras: ["camera_2", "camera_3"]
  - quadrant: "top_right"
    edge_threshold: 50.0
    target_cameras: ["camera_4", "camera_5"]
  # ... more rules ...
```

## Usage

1. Place homography matrices and exit rules in the appropriate camera directories
2. Update the paths in your configuration file to point to these files
3. The ML framework will use these files for:
   - Transforming coordinates between camera views
   - Determining when objects exit one camera and enter another
   - Evaluating tracking and Re-ID performance across cameras

## Notes

- Homography matrices should be computed using the same method as in the SpotOn backend
- Exit rules should match the physical layout of cameras in the environment
- File names and directory structure should be consistent with the configuration
- All paths in configuration files should be relative to this directory 