import torch
from scipy.sparse import lil_matrix


class TopK(object):
    def __init__(self, model, train_user_item_matrix):
        """
        Create a evaluator for recall@K evaluation
        :param model: the model we are going to evaluate
        :param train_user_item_matrix: the user-item pairs used in the training set. These pairs will be ignored
               in the recall calculation
        :param test_user_item_matrix: the held-out user-item pairs we make prediction against
        """
        self.model = model
        train_user_item_matrix = lil_matrix(train_user_item_matrix)
        self.train_user_item_matrix = train_user_item_matrix
        n_users = train_user_item_matrix.shape[0]

        if self.train_user_item_matrix is not None:
            self.user_to_train_set = {u: set(self.train_user_item_matrix.rows[u])
                                      for u in range(n_users) if self.train_user_item_matrix.rows[u]}
            self.max_train_count = max(len(row) for row in self.train_user_item_matrix.rows)
        else:
            self.max_train_count = 0

    def predict(self, users, items, artists, albums, k=500):
        """
        Compute the Top-K recall for a particular user given the predicted scores to items
        :param users: the users to eval the recall
        :param k: compute the recall for the top K items
        :return: recall@K
        """
        # compute the top (K +  Max Number Of Training Items for any user) items for each user

        item_scores, user_tops = torch.topk(self.model.item_scores(users, items, artists, albums,None, None, None ), k + self.max_train_count)
        
        item_scores = item_scores.cpu().numpy()                                                                                        
        results = []                                                                                                                   
        scores = []
        for user_id, tops, items in zip(users.cpu().numpy(), user_tops.cpu().numpy(), item_scores):                                    
            temp = []
            temp_scores = []
            train_set = self.user_to_train_set.get(user_id, set())                                                                     
            top_n_items = 0
            for j, i in enumerate(tops):
                # ignore item in the training set                                                                                      
                if i in train_set:                                                                                                     
                    continue                                                                                                           
                else:
                    temp.append(i)
                    temp_scores.append(items[j])                                                                                       
                top_n_items += 1
                if top_n_items == k:                                                                                                   
                    break
            results.append(temp)
            scores.append(temp_scores)                                                                                                 
        return results, scores


