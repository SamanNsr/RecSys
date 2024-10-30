import numpy as np
from scipy.sparse.linalg import svds


class SVDModelCF:
    def __init__(self, user_item_matrix, n_factors=10):
        self.user_item_matrix = user_item_matrix
        self.n_factors = n_factors
        self.user_factors = None
        self.sigma = None
        self.item_factors = None
        self.predicted_ratings = None

    def compute_svd(self):
        matrix = self.user_item_matrix.fillna(0).values
        user_ratings_mean = np.mean(matrix, axis=1)
        matrix_demeaned = matrix - user_ratings_mean.reshape(-1, 1)

        U, sigma, Vt = svds(matrix_demeaned, k=self.n_factors)
        self.user_factors = U
        self.sigma = np.diag(sigma)
        self.item_factors = Vt

        self.predicted_ratings = np.dot(
            np.dot(U, self.sigma), Vt) + user_ratings_mean.reshape(-1, 1)

    def predict_feedback(self, user_id):
        if self.predicted_ratings is None:
            raise ValueError(
                "SVD has not been computed. Call compute_svd() first.")

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
