import numpy as np
from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner


def main():

    # Read Video
    video_path = 'inputs/08fd33_4.mp4'
    frames, width, height, fps = read_video(video_path)

    # Initialize tracker
    model_path = r'yolov11l_finetune\weights\best.pt'
    tracker = Tracker(model_path)
    tracks = tracker.get_object_tracks(
        frames=frames[:450],
        read_from_stub=True,
        stub_path='stubs/track_stubs.pkl')

    # Interpolate ball position
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    # Assign Player Team
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                frame=frames[frame_num],
                bbox=track['bbox'],
                player_id=player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.teams_colors[team]

    # Assign Ball to closest Player
    ball_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = ball_assigner.assign_ball_to_player(
            player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(
                tracks['players'][frame_num][assigned_player]['team'])
        else:
            if not team_ball_control:
                # Default to 0 if no team has control yet
                team_ball_control.append(0)
            else:
                team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)
    # Draw Output
    # Draw Object Tracks
    output_frames = tracker.draw_annotations(
        video_frames=frames[:450],
        tracks=tracks,
        team_ball_control=team_ball_control
    )

    # Save Video
    output_path = 'outputs/out_vid.mp4'
    save_video(output_frames, width, height, fps, output_path)
    print("Output is Saved in outputs folder")


if __name__ == "__main__":
    main()
