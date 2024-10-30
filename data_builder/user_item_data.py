class UserItemData:
    def __init__(self, feedback_df):
        self.feedback_df = feedback_df
        self.user_item_matrix = None
    
    def build_user_item_matrix(self):
        self.user_item_matrix = self.feedback_df.pivot(index='user_id', columns='item_id', values='feedback')
        self.user_item_matrix = self.user_item_matrix.fillna(0)  # Fill NaNs with 0 (no feedback)
        return self.user_item_matrix
