# Dronebase Machine Learning Data Utils

## PUBLIC FACING

This package contains commonly used data functions for ML Engineers

```python
import ml_dronebase_data_utils as data_utils
...
```

# Object Detection Annotation Formatting
This package provides a Pascal VOC writer that renders ```*.xml``` annotation files for object detection tasks.
It supports regular object detection and oriented object detection annotations with an additional ```<angle>```_some angle_```</angle>``` parameter.

```python
from ml_dronebase_data_utils import PascalVOCWriter
writer = PascalVOCWriter()

for box in boxes:
    xmin, ymin, xmax, ymax, angle = box
    writer.addObject(
        name="some class name",
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
        angle=angle # Optional parameter
    )
writer.save(annotation_path)
```

# S3 Data Utils
This package also provides common AWS S3 data functions like downloading data, uploading data (data or trained models), train/test split, etc.

## Installation from source

Clone and ```cd``` into the root directory of this repo, then run the following:

```bash
pip install -e .
```

## Installation using pip

```bash
pip install ml-dronebase-data-utils
```
