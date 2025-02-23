# Video Surveillance Tracker

A video analysis system for object tracking and zone intrusion detection with configurable parameters.

## Features

- 🎯 Object tracking with history visualization
- 🚨 Zone intrusion detection and alarm triggering
- ⚙️ Fully configurable through YAML settings
- 📹 Video input/output processing
- 📊 Custom channel region configuration

## Configuration (config-v2.yaml)
```yaml
model_path: 'checkpoints/best.pt'
video_path: 'video/test2.mp4'
output_path: "outputs/output_custom_tracker_best/result.mp4"
channel_region:
  - [409, 1113]
  - [870, 1080]
  - [1745, 1500]
  - [630, 1520]
track_history_length: 30
alarm_threshold_seconds: 5
save_debug_frames: False
fps: null
```

### Key Parameters:
- `channel_region`: Polygon coordinates defining surveillance area
- `alarm_threshold_seconds`: Duration threshold for triggering alerts
- `track_history_length`: Number of frames to visualize tracking path
- `model_path`: Pretrained object detection model

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure settings in `model/config-v2.yaml`

3. Run main script:
```bash
python main.py --config model/config-v2.yaml
```

## License
MIT License
