import numpy as np
from sklearn.decomposition import NMF


class NMFModelCF:
    def __init__(self, user_item_matrix, n_factors=10, max_iter=200, random_state=42):
        self.user_item_matrix = user_item_matrix
        self.n_factors = n_factors
        self.max_iter = max_iter
        self.random_state = random_state
        self.user_factors = None
        self.item_factors = None
        self.predicted_ratings = None
        self.model = NMF(
            n_components=self.n_factors,
            max_iter=self.max_iter,
            random_state=self.random_state
        )

    def compute_nmf(self):
        matrix = self.user_item_matrix.fillna(0).values

        self.user_factors = self.model.fit_transform(matrix)
        self.item_factors = self.model.components_

        self.predicted_ratings = np.dot(self.user_factors, self.item_factors)

    def predict_feedback(self, user_id):
        if self.predicted_ratings is None:
            raise ValueError(
                "NMF has not been computed. Call compute_nmf() first.")

        user_index = self.user_item_matrix.index.get_loc(user_id)
        return self.predicted_ratings[user_index]

    def recommend_items(self, predicted_feedback, user_id, n_recomm=5):
        user_feedback = self.user_item_matrix.loc[user_id].values
        recommendations = [
            (item, feedback) for item, feedback in zip(self.user_item_matrix.columns, predicted_feedback)
            if user_feedback[self.user_item_matrix.columns.get_loc(item)] == 0
        ]
        recommendations.sort(key=lambda x: x[1], reverse=True)

        return recommendations[:n_recomm]
