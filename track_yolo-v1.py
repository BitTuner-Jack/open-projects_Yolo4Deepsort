import os
import tqdm
from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import time
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
from shapely.geometry import Point, Polygon
import logging
import yaml

# Configure logging 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    model_path: str
    video_path: str
    output_path: str
    channel_region: List[Tuple[int, int]]  # [(x1,y1), (x2,y2), ...]
    track_history_length: int = 30
    alarm_threshold_seconds: float = 10.0
    save_debug_frames: bool = False
    fps: Optional[float] = None


class ObjectTracker:
    def __init__(self, config: Config):
        self.config = config
        self.model = self._load_model()
        self.cap = self._init_video_capture()
        self.track_history = defaultdict(list)
        self.entry_times: Dict[int, float] = {}
        # create polygon object
        self.channel_polygon = Polygon(self.config.channel_region)

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.roi_mask = self._create_roi_mask()

        self.video_writer = self._init_video_writer()
        self.frame_count = 0

    def _load_model(self) -> YOLO:
        try:
            model = YOLO(self.config.model_path)
            logger.info(f"Model loaded successfully from {self.config.model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load mode: {str(e)}")
            raise

    def _init_video_capture(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.config.video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {self.config.video_path}")
        logger.info(f"Video caoture initialized: {self.config.video_path}")
        return cap
    
    
    def _create_roi_mask(self) -> np.ndarray:
        """
        Create ROI mask
        """
        
        mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        points = np.array(self.config.channel_region, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
        return cv2.merge([mask, mask, mask]) 

    def _init_video_writer(self) -> cv2.VideoWriter:
        fps = self.config.fps or self.cap.get(cv2.CAP_PROP_FPS)
        
        os.makedirs(os.path.dirname(self.config.output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            self.config.output_path,
            fourcc,
            fps,
            (self.frame_width, self.frame_height)
        )

        if not writer.isOpened():
            raise RuntimeError("Failed to create video writer")
        
        logger.info(f"Initialized video writer: {self.config.output_path}")
        return writer

    def _save_debug_frame(self, frame: np.ndarray, prefix: str = "debug"):
        if not self.config.save_debug_frames:
            return
        
        debug_dir = os.path.join(os.path.dirname(self.config.output_path), "debug_frames")
        os.makedirs(debug_dir, exist_ok=True)

        frame_path = os.path.join(debug_dir, f"{prefix}_frame_{self.frame_count:06d}.jpg")
        cv2.imwrite(frame_path, frame)

    def _is_in_channel(self, x: float, y: float, w: float, h: float) -> bool:
        """
        Check if the target is in the channel region
        Use the center point of the target for judgment
        """
        center_point = Point(x + w/2, y + h/2)
        # center_point = Point(x, y)
        return self.channel_polygon.contains(center_point)
    
    def _process_tracks(self, frame: np.ndarray, boxes: np.ndarray, track_ids: List[int],
                      class_ids: List[int]) -> np.ndarray:
        """Process and visualize tracking information"""
        current_time = time.time()
        time_in_channel = 0

        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            x, y, w, h = box
                
            # update track history
            track = self.track_history[track_id]
            track.append((float(x), float(y)))
            if len(track) > self.config.track_history_length:
                track.pop(0)

            # draw track line
            if len(track) > 1:
                points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], False, (230, 230, 230), 2)

            # check if in channel region
            if self._is_in_channel(x, y, w, h):
                if track_id not in self.entry_times:
                    # need to use video timestamp
                    self.entry_times[track_id] = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    logger.info(f"Object {track_id} entered channel region")
                else:
                    time_in_channel = (self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0) - self.entry_times[track_id]
                    if time_in_channel > self.config.alarm_threshold_seconds:
                        self._draw_alarm(frame, track_id, x, y)
                        logger.warning(f"Object {track_id} exceeded channel occupancy threshold")
            else:
                if track_id in self.entry_times:
                    logger.info(f"Object {track_id} left channel region")
                self.entry_times.pop(track_id, None)
            
            # draw detection box and stay time
            self._draw_detection_box(frame, track_id, x, y, w, h, time_in_channel, class_id)

        return frame
    
    def _draw_alarm(self, frame: np.ndarray, track_id: int, x: float, y: float):
        text = f"ALARM: Object {track_id} in channel > {self.config.alarm_threshold_seconds}s!"
        cv2.putText(frame, text, (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def _draw_detection_box(self, frame: np.ndarray, track_id: int, x: float, y: float, w: float, h: float,
                            time_in_channel: float, class_id: int):
        
        class_name = self.model.names[class_id]

        if class_name != "Other":
            # draw detection box
            cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), 
                        (int(x + w / 2), int(y + h / 2)), (255, 0, 0), 2)
            # display object ID and stay time
            if track_id in self.entry_times:
                cv2.putText(frame, f"{class_name} ID: {track_id}", (int(x - w / 2), int(y - h/2 - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(frame, f"Time: {int(time_in_channel)}s", (int(x - w / 2), int(y - h / 2 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    def _draw_channel_region(self, frame: np.ndarray) -> None:
        """
        Draw polygon channel region on frame
        """
        # convert to numpy array format for drawing
        points = np.array(self.config.channel_region, dtype=np.int32)
        # draw closed polygon
        cv2.polylines(frame, [points], True, (0, 255, 0), 2)
        # fill polygon (semi-transparent)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [points], (0, 255, 0))
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        

    def process_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        success, frame = self.cap.read()
        if not success:
            return False, None
        
        try:
            if self.config.save_debug_frames:
                self._save_debug_frame(frame, "frame")

            # masked_frame = frame * self.roi_mask
            masked_frame = cv2.bitwise_and(frame, self.roi_mask)

            if self.config.save_debug_frames:
                self._save_debug_frame(masked_frame, "masked")

            # Run YOLO tracking on the ROI
            results = self.model.track(masked_frame, persist=True, tracker="tracker_config_yolo.yaml")
            self._draw_channel_region(frame)
            if not results or not results[0].boxes:
                self.video_writer.write(frame)
                self.frame_count += 1
                return True, frame
            
            # Get detection results and map to orginal frame
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id
            class_ids = results[0].boxes.cls.cpu().numpy()
            if track_ids is None:
                self.video_writer.write(frame)
                self.frame_count += 1
                return True, frame
            track_ids = track_ids.int().cpu().numpy().tolist()


            processed_frame = self._process_tracks(frame, boxes, track_ids, class_ids)
            self.video_writer.write(processed_frame)

            if self.config.save_debug_frames:
                self._save_debug_frame(processed_frame, "processed")
            
            self.frame_count += 1

            return True, processed_frame

        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return False, None

    def cleanup(self):
        """Cleanup resource"""
        self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Cleanup completed")

    def run(self):
        """Main processing loop"""
        try:
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            start_time = time.time()

            with tqdm.tqdm(total=total_frames, desc="Processing frames") as pbar:
                while True:
                    success, _ = self.process_frame()
                    if not success:
                        break
                    pbar.update(1)

            end_time = time.time()
            processing_time = end_time - start_time
            fps = self.frame_count / processing_time

            logger.info(f"Processing completed:")
            logger.info(f"Total frames: {self.frame_count}")
            logger.info(f"Processing time: {processing_time:.2f} seconds")
            logger.info(f"Average FPS: {fps:.2f}")

        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        self.cap.release()
        if self.video_writer:
            self.video_writer.release()
        logger.info(f"Processed {self.frame_count} frames")
        logger.info(f"Output saved to: {self.config.output_path}")

def load_config(config_path: str) -> Config:
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            return Config(**config_dict)
    except Exception as e:
        logger.error(f"Failed to load config: {str(e)}")
        raise

def main():
    try:
        config = load_config('config.yaml')
        tracker = ObjectTracker(config)
        tracker.run()
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
