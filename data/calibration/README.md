# Camera calibration

`camera_calibration.example.yaml` documents the expected schema with sample
values from one 640×480 camera. It is not a reusable calibration.

Create the local runtime file:

```bash
cp data/calibration/camera_calibration.example.yaml \
   data/calibration/camera_calibration.yaml
```

Then replace the example values using the procedures in
[`docs/calibration.md`](../../docs/calibration.md). The runtime filename is
ignored by Git because calibration is specific to a camera, resolution, mount,
and workcell.
