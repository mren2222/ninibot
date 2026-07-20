# Ninibot portfolio media plan

This is the shot list for turning the repository into a convincing portfolio
page. Add the P0 items first; the README already reserves their locations.

## P0 — essential

| README location | File / destination | What to capture | Format |
| --- | --- | --- | --- |
| Directly below the title and badges | `media/hero-workcell.jpg` | Clean three-quarter view showing the SO-101, wrist camera, cup, ArUco target, and dispenser/button in one uncluttered frame. Use a readable mid-motion arm pose. | 16:9, at least 1600×900, JPG, ideally under 1 MB |
| “Quick demo” | `media/demo-preview.gif` | One continuous highlight: cup detection overlay → gripper closes → cup is carried → button press. Cut dead time and use no more than three tiny captions. | 8–12 s, 960×540, 10–15 fps, looped, ideally under 8 MB |
| Under the GIF | `media/demo-thumbnail.jpg` linked to the full video | Choose the clearest frame from the full demo and add a simple play icon. | 16:9, 1280×720 |
| Link from “Quick demo” | YouTube video or GitHub Release asset | One complete successful run with a wide workcell view plus short camera/RViz inserts. Show target scan, grasp, transfer, press, and final state without hiding interventions. | 45–90 s, 1080p, H.264 MP4; captions recommended |

Suggested full-demo edit:

1. 0–4 s: workcell overview and one-sentence goal.
2. 4–15 s: camera overlay finds the cup and scans the target marker.
3. 15–30 s: visual alignment and grasp.
4. 30–50 s: lift, carry, and placement.
5. 50–65 s: marker alignment and button press.
6. 65–75 s: `/task_manager_status` reaches `DONE` and the final outcome.
7. 75–90 s: optional synchronized hardware/RViz view.

## P1 — strong supporting evidence

| README / docs location | Filename | What to show | Format |
| --- | --- | --- | --- |
| Manipulation explanation | `media/grasp-stages.png` | A 2×2 sequence labeled `SEARCHING`, `TRACKING_CENTER`, `FINE_ALIGN`, and `FINISHED`. Preserve the detection box and actual status overlay. | About 1600×1000 |
| Placement and press explanation | `media/aruco-place-and-press.png` | Left: destination marker lock and cup placement. Right: button marker and end-effector alignment. | Horizontal comparison, about 1600×900 |
| Perception explanation | `media/perception-comparison.png` | The same camera frame with HSV, YOLO, and hybrid output. Show behavioral differences without claiming an accuracy improvement. | Three columns, about 1800×800 |
| “First launch: RViz / dry-run” | `media/rviz-digital-twin.png` | Robot model, TF frames, target joints, and any path/ghost visualization. Crop unrelated terminals and panels. | 16:9, 1600×900 |
| Hardware setup | `media/hardware-annotated.jpg` | Label the wrist camera, gripper, serial control board, cup, placement marker, and button marker. | 3:2, at least 1500 px wide |

The existing
`data/trajectories/fine_align_prep/sessions/2026-06-15_8cm_base_raw/raw_vs_processed_comparison.png`
is already embedded in the README as quantitative engineering evidence. The
companion `processed_joint_dynamics.png` is better kept in the detailed sample
directory rather than used as a homepage result.

## P2 — experimental and engineering depth

| Destination | Filename | What to capture |
| --- | --- | --- |
| Experimental learning section | `media/policy-preview.gif` | RViz real-arm state and policy ghost side by side. Label it “preview only — does not command hardware.” |
| Experimental audio section | `media/water-flow-monitor.png` | Score/baseline and start/stop thresholds as water begins and ends. Label it “standalone experiment.” |
| Limitations / recovery | `media/recovery-demo.gif` | A genuinely repeatable case where the cup is briefly lost, the controller searches, and it reacquires. Add only after it works consistently. |
| `docs/calibration.md` | `media/calibration-setup.jpg` | Checkerboard, camera viewpoint, ArUco dimensions, and physical layout used for calibration. |
| Longer demo or release notes | `media/rviz-hardware-split.mp4` | Synchronized real arm and RViz showing command/feedback correspondence. |

## Capture rules

- Keep the cup, marker layout, lighting, and background consistent across the
  hero, GIF, and full video.
- Record the wide shot on a tripod and keep the full arm envelope visible.
- Record camera and RViz output directly instead of filming a monitor.
- Overlay real state values such as `/task_manager_status` or
  `/cup_servo_status`; they are stronger evidence than promotional captions.
- Avoid faces, notifications, addresses, device serial numbers, and private
  screens in the frame.
- Label dry-run footage as dry-run and learned-policy footage as preview only.
- Present the water-flow monitor as an experimental standalone subsystem until
  it is actually connected to the task manager.
- Do not describe one successful run as a success rate, and do not claim
  accuracy, robustness, generalization, or collision safety without a measured
  benchmark.

## README insertion snippets

Replace the HTML comment near the top of `README.md` with:

```markdown
![Ninibot workcell](media/hero-workcell.jpg)
```

For the short loop, replace the comment under “Quick demo” with:

```markdown
[![Ninibot demo loop](media/demo-preview.gif)](FULL_DEMO_URL)
```

Replace `FULL_DEMO_URL` with the YouTube or GitHub Release URL. A linked GIF is
more reliable in a GitHub README than committing and trying to embed a large
MP4 directly.

