import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ItemBasedMemoryCF:
    def __init__(self, user_item_matrix, similarity_metric='cosine'):
        self.user_item_matrix = user_item_matrix
        self.similarity_metric = similarity_metric
        self.item_similarity_matrix = None
    
    def compute_item_similarity(self):
        if self.similarity_metric == 'cosine':
            self.item_similarity_matrix = cosine_similarity(self.user_item_matrix.T)
        elif self.similarity_metric == 'pearson':
            self.item_similarity_matrix = self.user_item_matrix.corr(method='pearson').to_numpy()
        else:
            raise ValueError("Unsupported similarity metric. Use 'cosine' or 'pearson'.")

        np.fill_diagonal(self.item_similarity_matrix, 0)

    def predict_feedback(self, user_id):
        user_feedback_vector = self.user_item_matrix.loc[user_id]
        weighted_feedback = np.dot(self.item_similarity_matrix, user_feedback_vector.fillna(0))
        normalization = np.abs(self.item_similarity_matrix).sum(axis=1)
        
        predicted_feedback = np.divide(weighted_feedback, normalization, out=np.zeros_like(weighted_feedback), where=normalization != 0)
        return predicted_feedback

    def recommend_items(self, predicted_feedback, user_id, n_recomm=5):
        user_feedback = self.user_item_matrix.loc[user_id].values
        recommendations = [
            (item, feedback) for item, feedback in zip(self.user_item_matrix.columns, predicted_feedback)
            if user_feedback[self.user_item_matrix.columns.get_loc(item)] == 0
        ]
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:n_recomm]
