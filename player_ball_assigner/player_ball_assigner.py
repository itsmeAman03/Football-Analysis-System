import sys
from utils import get_center_of_bbox
from utils.bbox_utils import measure_distance
sys.path.append('../')


class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 70

    def assign_ball_to_player(self, players, ball_bbox):

        ball_position = get_center_of_bbox(ball_bbox)

        minimum_dist = 999999
        assigned_player = -1

        for player_id, player in players.items():

            player_bbox = player['bbox']

            distance_left = measure_distance(
                (player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_distance(
                (player_bbox[2], player_bbox[-1]), ball_position)

            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance:
                if distance < minimum_dist:
                    assigned_player = player_id
                    minimum_dist = distance

        return assigned_player
