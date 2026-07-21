# Calibration and diagnostics

Ninibot uses three distinct forms of calibration:

1. Camera intrinsics for metric pose estimation.
2. Camera-to-workcell geometry for ArUco-guided manipulation.
3. SO-101 motor calibration for the physical arm.

The public repository includes
`data/calibration/camera_calibration.example.yaml` only as a schema example.
Generate a machine-specific `camera_calibration.yaml` before relying on metric
camera measurements.

## 1. Camera intrinsics

Generate a checkerboard:

```bash
python3 tools/calibration/generate_chessboard.py
```

Capture calibration images from varied distances, positions, and orientations:

```bash
python3 tools/calibration/capture_chessboard.py \
  --camera-id 0 \
  --pattern 8x6 \
  --save-dir data/calibration/images
```

Estimate the camera matrix and distortion coefficients:

```bash
python3 tools/calibration/calibrate_camera.py \
  --image-dir data/calibration/images \
  --output data/calibration/camera_calibration.yaml \
  --pattern 8x6 \
  --square-size 0.028
```

The square size is expressed in metres and must match the printed checkerboard.

Inspect the estimated intrinsics with a known ArUco marker:

```bash
python3 tools/calibration/check_intrinsics.py \
  --camera-id 0 \
  --calibration data/calibration/camera_calibration.yaml \
  --marker-length 0.05 \
  --target-marker-id 0
```

The marker length is also expressed in metres and must match the physical print.

## 2. ArUco geometry checks

Inspect a marker pose in the camera optical frame:

```bash
python3 tools/calibration/check_aruco_camera.py \
  --camera-id 0 \
  --calibration data/calibration/camera_calibration.yaml \
  --marker-length 0.05 \
  --target-marker-id 0
```

Check the camera-to-base extrinsic transform:

```bash
python3 tools/calibration/check_aruco_extrinsics.py \
  --camera-id 0 \
  --calibration data/calibration/camera_calibration.yaml \
  --marker-length 0.05 \
  --target-marker-id 0
```

Keep the marker dictionary, marker ID, printed dimensions, and coordinate-frame
conventions consistent across calibration and runtime launches.

## 3. SO-101 motor calibration

Motor calibration is independent of camera calibration. Recalibrating an arm
that already works can invalidate known joint offsets, so back up the existing
LeRobot calibration and confirm the serial port before proceeding.

```bash
python3 tools/calibration/so_follower_calibration.py \
  --port /dev/ttyACM0 \
  --robot so101 \
  --mode calibrate
```

Run this only with the workspace clear and the arm supported as required by the
SO-101 calibration procedure.

## 4. Perception smoke tests

Test HSV cup segmentation:

```bash
python3 tools/debug/detect_cup.py --camera-id 0
```

Test YOLO cup detection with user-supplied weights:

```bash
python3 tools/debug/yolo_coco_cup_test.py \
  --model /absolute/path/to/yolo11n.pt \
  --camera-id 0
```

YOLO weights are intentionally not committed to this public repository.

Test desktop ArUco detection and homography:

```bash
python3 tools/debug/aruco_detect_test.py --camera-id 0
```

These utilities validate individual perception components. They do not validate
the complete arm trajectory or establish collision safety.
