from sklearn.cluster import KMeans


class KMeansModelCF:
    def __init__(self, user_item_matrix, n_clusters=10, random_state=42):
        self.user_item_matrix = user_item_matrix
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = KMeans(n_clusters=self.n_clusters,
                            random_state=self.random_state)
        self.user_clusters = None
        self.cluster_centers = None

    def compute_kmeans(self):
        matrix = self.user_item_matrix.fillna(0).values

        self.user_clusters = self.model.fit_predict(matrix)
        self.cluster_centers = self.model.cluster_centers_

    def predict_feedback(self, user_id):
        if self.user_clusters is None:
            raise ValueError(
                "KMeans has not been computed. Call fit_kmeans() first.")

        user_index = self.user_item_matrix.index.get_loc(user_id)
        user_cluster = self.user_clusters[user_index]

        return self.cluster_centers[user_cluster]

    def recommend_items(self, predicted_feedback, user_id, n_recomm=5):
        user_feedback = self.user_item_matrix.loc[user_id].values
        recommendations = [
            (item, feedback) for item, feedback in zip(self.user_item_matrix.columns, predicted_feedback)
            if user_feedback[self.user_item_matrix.columns.get_loc(item)] == 0
        ]
        recommendations.sort(key=lambda x: x[1], reverse=True)

        return recommendations[:n_recomm]
