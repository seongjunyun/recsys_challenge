import torch
from scipy.sparse import lil_matrix
import numpy as np

class RecallEvaluator(object):
    def __init__(self, model, train_user_item_matrix, test_user_item_matrix):
        """
        Create a evaluator for recall@K evaluation
        :param model: the model we are going to evaluate
        :param train_user_item_matrix: the user-item pairs used in the training set. These pairs will be ignored
               in the recall calculation
        :param test_user_item_matrix: the held-out user-item pairs we make prediction against
        """
        self.model = model
        self.train_user_item_matrix = train_user_item_matrix
        self.test_user_item_matrix = test_user_item_matrix
        n_users = train_user_item_matrix.shape[0]
        self.user_to_test_set = {u: set(self.test_user_item_matrix.rows[u])
                                 for u in range(n_users) if self.test_user_item_matrix.rows[u]}

        if self.train_user_item_matrix is not None:
            self.user_to_train_set = {u: set(self.train_user_item_matrix.rows[u])
                                      for u in range(n_users) if self.train_user_item_matrix.rows[u]}
            self.max_train_count = max(len(row) for row in self.train_user_item_matrix.rows)
        else:
            self.max_train_count = 0

    def dcg(self, relevant_elements, retrieved_elements, k):
        """Compute the Discounted Cumulative Gain.
        Rewards elements being retrieved in descending order of relevance.
        \[ DCG = rel_1 + \sum_{i=2}^{|R|} \frac{rel_i}{\log_2(i + 1)} \]
        Args:
            retrieved_elements (list): List of retrieved elements
            relevant_elements (list): List of relevant elements
            k (int): 1-based index of the maximum element in retrieved_elements
            taken in the computation
        Note: The vector `retrieved_elements` is truncated at first, THEN
        deduplication is done, keeping only the first occurence of each element.
        Returns:
            DCG value
        """
        retrieved_elements = retrieved_elements[:k]
        relevant_elements = relevant_elements
        if len(retrieved_elements) == 0 or len(relevant_elements) == 0:
            return 0.0
        # Computes an ordered vector of 1.0 and 0.0
        score = [float(el in relevant_elements) for el in retrieved_elements]
        # return score[0] + np.sum(score[1:] / np.log2(
        #     1 + np.arange(2, len(score) + 1)))
        return np.sum(score / np.log2(1 + np.arange(1, len(score) + 1)))


    def ndcg(self, relevant_elements, retrieved_elements, k):
        r"""Compute the Normalized Discounted Cumulative Gain.
        Rewards elements being retrieved in descending order of relevance.
        The metric is determined by calculating the DCG and dividing it by the
        ideal or optimal DCG in the case that all recommended tracks are relevant.
        Note:
        The ideal DCG or IDCG is on our case equal to:
        \[ IDCG = 1+\sum_{i=2}^{min(\left| G \right|, k)}\frac{1}{\log_2(i +1)}\]
        If the size of the set intersection of \( G \) and \( R \), is empty, then
        the IDCG is equal to 0. The NDCG metric is now calculated as:
        \[ NDCG = \frac{DCG}{IDCG + \delta} \]
        with \( \delta \) a (very) small constant.
        The vector `retrieved_elements` is truncated at first, THEN
        deduplication is done, keeping only the first occurence of each element.
        Args:
            retrieved_elements (list): List of retrieved elements
            relevant_elements (list): List of relevant elements
            k (int): 1-based index of the maximum element in retrieved_elements
            taken in the computation
        Returns:
            NDCG value
        """

        idcg = self.dcg(
            relevant_elements, relevant_elements, min(k, len(relevant_elements)))
        if idcg == 0:
            raise ValueError("relevent_elements is empty, the metric is"
                             "not defined")
        true_dcg = self.dcg(relevant_elements, retrieved_elements, k)
        return true_dcg / idcg



    def eval(self, users, items, artists, albums, titles, titles_len, ratios, k=50):
        """
        Compute the Top-K recall for a particular user given the predicted scores to items
        :param users: the users to eval the recall
        :param k: compute the recall for the top K items
        :return: recall@K
        """
        # compute the top (K +  Max Number Of Training Items for any user) items for each user

        
        _, user_tops = torch.topk(self.model.item_scores(users, items, artists, albums, titles, titles_len, ratios), k + self.max_train_count)
        
        user_tops = items[user_tops] 
        recalls = []
        ndcgs = []
        for user_id, tops in zip(users.cpu().numpy(), user_tops.cpu().numpy()):
            temp = []
            train_set = self.user_to_train_set.get(user_id, set())
            test_set = self.user_to_test_set.get(user_id, set())            
            top_n_items = 0
            hits = 0
            #k = len(test_set)
            for i in tops:
                # ignore item in the training set
                if i in train_set:
                    continue
                elif i in test_set:
                    hits += 1
                temp.append(i)
                top_n_items += 1
                if top_n_items == k:
                    break

            ndcg_score = self.ndcg(list(test_set), temp, k) 
            recalls.append(hits / float(len(test_set)))
            ndcgs.append(ndcg_score)

        
        return recalls, ndcgs



    def predict(self, sess, users, k=500):
        """
        Compute the Top-K recall for a particular user given the predicted scores to items
        :param users: the users to eval the recall
        :param k: compute the recall for the top K items
        :return: recall@K
        """
        # compute the top (K +  Max Number Of Training Items for any user) items for each user

        _, user_tops = sess.run(tf.nn.top_k(self.model.item_scores, k + self.max_train_count),
                                {self.model.score_user_ids: users})
        results = []
        for user_id, tops in zip(users, user_tops):
            train_set = self.user_to_train_set.get(user_id, set())
            test_set = self.user_to_test_set.get(user_id, set())
            top_n_items = 0
            for i in tops:
                # ignore item in the training set
                if i in train_set:
                    continue
                else:
                    results.append(i)
                top_n_items += 1
                if top_n_items == k:
                    break

        return results
