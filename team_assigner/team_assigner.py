from sklearn.cluster import KMeans
from typing import List


class TeamAssigner:
    def __init__(self):
        self.teams_colors = {}
        self.player_team_dict = {}

    def get_clustering_model(self, top_panel):
        # Reshare the image to 2D array
        img_2d = top_panel.reshape((-1, 3))

        # Perform K-Means cluster with 2 cluster
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1)
        kmeans.fit(img_2d)

        return kmeans

    def get_player_team_color(self, frame, bbox: List[float]):
        # Crop the Image
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        # Take Top half of the image to exract the jersey color
        top_panel = image[0:int(image.shape[0]/2), :]

        if top_panel.shape[0] == 0:
            return None

        kmeans = self.get_clustering_model(top_panel)

        # Get cluster labels
        labels = kmeans.labels_

        # Reshape the labels to the original image shape
        segmented_image = labels.reshape(
            top_panel.shape[0], top_panel.shape[1])

        # Corner Cluster
        corner_clusters = [segmented_image[0, 0], segmented_image[0, -1],
                           segmented_image[-1, 0], segmented_image[-1, -1]]
        non_player_cluster = max(set(corner_clusters),
                                 key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        color = kmeans.cluster_centers_[player_cluster]

        return color

    def assign_team_color(self, frame, player_detections):

        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection['bbox']
            player_color = self.get_player_team_color(frame, bbox)
            if player_color is not None:
                player_colors.append(player_color)

        if not player_colors:
            return None

        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.teams_colors[1] = self.kmeans.cluster_centers_[0]
        self.teams_colors[2] = self.kmeans.cluster_centers_[1]

    def get_player_team(self, frame, bbox, player_id):

        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_team_color(frame, bbox)
        team_id = 0
        if player_color is not None:
            team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
            team_id += 1

        self.player_team_dict[player_id] = team_id

        return team_id


if __name__ == "__main__":
    print("=====> Team Assigner <=====")
