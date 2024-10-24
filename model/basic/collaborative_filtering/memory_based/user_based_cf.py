import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class UserBasedMemoryCF:
    def __init__(self, user_item_matrix, similarity_metric='cosine'):
        self.user_item_matrix = user_item_matrix
        self.similarity_metric = similarity_metric
        self.user_similarity_matrix = None

    def compute_user_similarity(self):
        if self.similarity_metric == 'cosine':
            self.user_similarity_matrix = cosine_similarity(self.user_item_matrix)
        elif self.similarity_metric == 'pearson':
            self.user_similarity_matrix = np.corrcoef(self.user_item_matrix)
        else:
            raise ValueError("Unsupported similarity metric.")
        print("User similarity matrix computed.")
    
    def predict_feedback(self, user_id):
        if self.user_similarity_matrix is None:
            self.compute_user_similarity()

        user_feedback_vector = self.user_item_matrix.iloc[user_id - 1].values

        weighted_sum = np.dot(self.user_similarity_matrix[user_id - 1], self.user_item_matrix)

        similarity_sum = np.sum(self.user_similarity_matrix[user_id - 1])

        predicted_feedback = weighted_sum / (similarity_sum + 1e-8)

        predicted_feedback[user_feedback_vector > 0] = 0

        return predicted_feedback

    def recommend_items(self, predicted_feedback, user_id, n_recomm=5):
        user_feedback = self.user_item_matrix.iloc[user_id - 1].values
        
        non_interacted_items = [
            (item, feedback) for item, feedback in zip(self.user_item_matrix.columns, predicted_feedback)
            if user_feedback[self.user_item_matrix.columns.get_loc(item)] == 0
        ]
        
        non_interacted_items.sort(key=lambda x: x[1], reverse=True)
        
        return non_interacted_items[:n_recomm]
