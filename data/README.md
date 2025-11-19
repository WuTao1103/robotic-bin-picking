# Data Directory

This directory should contain your RGB-D dataset for bin-picking.

## Required Files

Place the following files in this directory:

```
data/
├── Color_3840x2160.png      # RGB image
├── pcd.ply                   # Point cloud (PLY format)
├── rgb_intrinsic.yaml        # RGB camera intrinsic parameters
├── depth_intrinsic.yaml      # Depth camera intrinsic parameters
└── rgb_distortion.yaml       # RGB camera distortion coefficients
```

## File Formats

### Camera Intrinsic YAML Format

```yaml
fx: 1234.56
fy: 1234.56
cx: 960.0
cy: 540.0
width: 1920
height: 1080
```

### Distortion YAML Format

```yaml
k1: 0.0
k2: 0.0
p1: 0.0
p2: 0.0
k3: 0.0
```

## Data Sources

You can obtain compatible datasets from:
- Intel RealSense cameras (D435, D455, L515)
- Azure Kinect DK
- Custom RGB-D sensors

## Notes

- Large data files (images, point clouds) are excluded from git by default
- YAML configuration files are included in version control
- Make sure your point cloud is in millimeters (mm) units
