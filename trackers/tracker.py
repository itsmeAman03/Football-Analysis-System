from utils import get_bbox_width, get_center_of_bbox
import pickle
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import os
import sys
import pandas as pd
sys.path.append('../')


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.BATCH_SIZE = 20
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        detections = []
        for i in range(0, len(frames), self.BATCH_SIZE):
            detections_batch = self.model.predict(
                frames[i:i+self.BATCH_SIZE], conf=0.1)
            detections.extend(detections_batch)
            print(
                f"===============> Processed {i+self.BATCH_SIZE} / {len(frames)} frames <===============")
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        tracks = None
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
        detections = self.detect_frames(frames)

        tracks = {
            'players': [],
            'referees': [],
            'ball': []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Convert to supervision Detection Format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # convert goalkeeper to player object
            if detection_supervision.class_id is not None:
                for obj_index, class_id in enumerate(detection_supervision.class_id):
                    if cls_names[class_id] == 'goalkeeper':
                        detection_supervision.class_id[obj_index] = cls_names_inv['player']

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(
                detection_supervision)

            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks['players'][frame_num][track_id] = {'bbox': bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks['referees'][frame_num][track_id] = {'bbox': bbox}

                if cls_id == cls_names_inv['ball']:
                    # Assuming there's only one ball per frame, track_id is not strictly needed
                    tracks['ball'][frame_num][1] = {'bbox': bbox}
                    print(f"Ball detected in frame {frame_num} at bbox {bbox}")

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def interpolate_ball_positions(self, ball_positions):
        # Extract bboxes or None for each frame
        bboxes = []
        for x in ball_positions:
            bbox = x.get(1, {}).get('bbox')

            if bbox:
                bboxes.append(bbox)
            else:
                bboxes.append([np.nan, np.nan, np.nan, np.nan])

        df_ball_positions = pd.DataFrame(
            bboxes, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate Missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [
            {1: {'bbox': x}}
            for x in df_ball_positions.to_numpy().tolist()
        ]

        return ball_positions

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(frame,
                    center=(int(x_center), y2),
                    axes=(int(width), int(0.35*width)),
                    angle=0,
                    startAngle=-45,
                    endAngle=235,
                    color=color,
                    thickness=2)
        # Optionally add the track ID as text

        rectangle_width = 40
        rectangle_height = 20

        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2 - rectangle_height//2) + 15
        y2_rect = (y2 + rectangle_height//2) + 15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED
                          - 1)
            x1_text = x1_rect + 12
            if track_id > 99:
                track_id -= 10

            cv2.putText(frame,
                        f"{track_id}",
                        (int(x1_text), int(y1_rect + 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),
                        2)
        return frame

    def draw_triangle(self, frame, bbox, color):
        try:
            if np.any(np.isnan(bbox)):
                return frame

            y = int(bbox[1])
            x, _ = get_center_of_bbox(bbox)
            if x is None:
                return frame

            # Convert all points to integers
            triangle_points = np.array([
                [int(x), int(y)],
                [int(x-10), int(y-20)],
                [int(x+10), int(y-20)]
            ], dtype=np.int32)  # Ensure int32 dtype for OpenCV

            # Reshape for OpenCV contours format
            triangle_points = triangle_points.reshape((-1, 1, 2))

            cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
            cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

            return frame
        except Exception as e:
            print(f"Error drawing triangle: {e}")
            return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Draw Semi Transparent Rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970),
                      (255, 255, 255), cv2.FILLED)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]

        # Get Number od time each team had the ball
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[
            0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[
            0]

        # % Each team had ball
        total_frames_with_ball = team_1_num_frames + team_2_num_frames
        if total_frames_with_ball == 0:
            team_1 = 0
            team_2 = 0
        else:
            team_1 = (team_1_num_frames/total_frames_with_ball)*100
            team_2 = (team_2_num_frames/total_frames_with_ball)*100

        cv2.putText(frame, f'Team 1 Ball Control: {team_1:.2f}%', (
            1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f'Team 2 Ball Control: {team_2:.2f}%', (
            1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks['players'][frame_num]
            referee_dict = tracks['referees'][frame_num]
            ball_dict = tracks['ball'][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(
                    frame=frame,
                    bbox=player['bbox'],
                    color=color,
                    track_id=track_id)

                if player.get("has_ball", False):
                    frame = self.draw_triangle(
                        frame, player['bbox'], (0, 0, 255))
            # Draw Referees
            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(
                    frame=frame,
                    bbox=referee['bbox'],
                    color=(0, 255, 255),  # Yellow for referees
                    track_id=track_id
                )

            # Draw Ball
            for _, ball in ball_dict.items():
                frame = self.draw_triangle(
                    frame=frame,
                    bbox=ball['bbox'],
                    color=(0, 255, 0),  # Green for the ball
                )

            # Draw Team Ball Control
            frame = self.draw_team_ball_control(
                frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames
