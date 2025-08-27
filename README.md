# OpenCV Panorama Stitcher (Pro)

A clean ORB + homography panorama stitcher with CLI.

## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
python scripts/stitch.py --folder docs/example_pano --save outputs/panorama.jpg
```
