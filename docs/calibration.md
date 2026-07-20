# Calibration

## 1. 相机内参

生成棋盘格:

```bash
python3 tools/calibration/generate_chessboard.py
```

采集图片:

```bash
python3 tools/calibration/capture_chessboard.py --camera-id 0 --pattern 8x6 --save-dir data/calibration/images
```

计算内参:

```bash
python3 tools/calibration/calibrate_camera.py --image-dir data/calibration/images --output data/calibration/camera_calibration.yaml --pattern 8x6 --square-size 0.028
```

检查内参:

```bash
python3 tools/calibration/check_intrinsics.py --camera-id 0 --calibration data/calibration/camera_calibration.yaml --marker-length 0.05 --target-marker-id 0
```

## 2. ArUco 相机检查

查看 marker 在 camera optical frame 下的位置:

```bash
python3 tools/calibration/check_aruco_camera.py --camera-id 0 --calibration data/calibration/camera_calibration.yaml --marker-length 0.05 --target-marker-id 0
```

检查 camera -> base extrinsics:

```bash
python3 tools/calibration/check_aruco_extrinsics.py --camera-id 0 --calibration data/calibration/camera_calibration.yaml --marker-length 0.05 --target-marker-id 0
```

## 3. SO101 电机 calibration

SO101 电机 calibration 和相机标定不同。已经能正常控制时谨慎使用:

```bash
python3 tools/calibration/so_follower_calibration.py --port /dev/ttyACM0 --robot so101 --mode calibrate
```

## 4. Debug smoke tests

HSV cup:

```bash
python3 tools/debug/detect_cup.py --camera-id 0
```

YOLO cup:

```bash
python3 tools/debug/yolo_coco_cup_test.py --model data/models/yolo/yolo11n.pt --camera-id 0
```

桌面 ArUco homography:

```bash
python3 tools/debug/aruco_detect_test.py --camera-id 0
```
