from collections import defaultdict

import numpy as np
from scipy.sparse import dok_matrix, lil_matrix
from tqdm import tqdm
import pickle

def citeulike(tag_occurence_thres=10):
    user_dict = defaultdict(set)
    for u, item_list in enumerate(open("citeulike-t/users.dat").readlines()):
        items = item_list.strip().split(" ")
        # ignore the first element in each line, which is the number of items the user liked. 
        for item in items[1:]:
            user_dict[u].add(int(item))

    n_users = len(user_dict)
    n_items = max([item for items in user_dict.values() for item in items]) + 1

    user_item_matrix = dok_matrix((n_users, n_items), dtype=np.int32)
    for u, item_list in enumerate(open("citeulike-t/users.dat").readlines()):
        items = item_list.strip().split(" ")
        # ignore the first element in each line, which is the number of items the user liked. 
        for item in items[1:]:
            user_item_matrix[u, int(item)] = 1

    n_features = 0
    for l in open("citeulike-t/tag-item.dat").readlines():
        items = l.strip().split(" ")
        if len(items) >= tag_occurence_thres:
            n_features += 1
    print("{} features over tag_occurence_thres ({})".format(n_features, tag_occurence_thres))
    features = dok_matrix((n_items, n_features), dtype=np.int32)
    feature_index = 0
    for l in open("citeulike-t/tag-item.dat").readlines():
        items = l.strip().split(" ")
        if len(items) >= tag_occurence_thres:
            features[[int(i) for i in items], feature_index] = 1
            feature_index += 1

    return user_item_matrix, features

def spotify(mode, valid):
   
    with open('/home/yseongjun/recsys/data/dic/song2artist_filtered.pickle','rb') as f:
        artists_dic = pickle.load(f)
    with open('/home/yseongjun/recsys/data/dic/song2album_filtered.pickle','rb') as f:
        albums_dic = pickle.load(f)
    with open('/home/yseongjun/recsys/data/title/playlist_title_pad.pickle','rb') as f:
        titles = pickle.load(f)
    with open('/home/yseongjun/recsys/data/title/title_len.pickle','rb') as f:
        titles_len = pickle.load(f)
    with open('/home/yseongjun/recsys/data/ratios.pickle','rb') as f:
        ratios = pickle.load(f)
    with open('/home/yseongjun/recsys/data/dic/unigram_song.pickle','rb') as f:
        unigram_probs = pickle.load(f)

    if mode == 'train':

        if valid == 'True':
            with open('/home/yseongjun/recsys/data/train_valid/train_matrix_filtered_one.pickle','rb') as f:
                train = pickle.load(f)
            with open('/home/yseongjun/recsys/data/train_valid/valid_candidates_2000.pickle','rb') as f:
                val_candidates = pickle.load(f)            
            with open('/home/yseongjun/recsys/data/train_valid/valid_matrix_filtered.pickle','rb') as f:
                valid = pickle.load(f)
        

            return train, valid, None, artists_dic, albums_dic, titles, titles_len, ratios, val_candidates, unigram_probs
        
    elif mode == 'valid':
        with open('/home/yseongjun/recsys/data/train_valid/train_matrix_filtered_one.pickle','rb') as f:
            train = pickle.load(f)
        with open('/home/yseongjun/recsys/data/train_valid/valid_candidates_2000.pickle','rb') as f:
            val_candidates = pickle.load(f)            
        with open('/home/yseongjun/recsys/data/train_valid/valid_matrix_filtered.pickle','rb') as f:
            valid = pickle.load(f)
        

        return train, valid, None, artists_dic, albums_dic, titles, titles_len, ratios, val_candidates, unigram_probs
 
    else:
        with open('/home/yseongjun/recsys/data/train_valid/train_matrix_filtered_one.pickle','rb') as f:
            matrix = pickle.load(f)
        #matrix = matrix.todok()
        return matrix, None, None, artists_dic, albums_dic, titles, titles_len, ratios, None, unigram_probs

def split_data(user_item_matrix, split_ratio=(9, 1, 0), seed=1):
    # set the seed to have deterministic results
    np.random.seed(seed)
    train = dok_matrix(user_item_matrix.shape)
    validation = dok_matrix(user_item_matrix.shape)
    # convert it to lil format for fast row access
    user_item_matrix = lil_matrix(user_item_matrix)
    for user in tqdm(range(user_item_matrix.shape[0]) , desc="Split data into train/valid/test"):
        items = list(user_item_matrix.rows[user])
        #np.random.shuffle(items)
        if user >= 1007000 and user < 1009000:
            train_count = int(len(items) * split_ratio[0] / sum(split_ratio))
            valid_count = int(len(items) * split_ratio[1] / sum(split_ratio))
        else:
            train_count = len(items)
            valid_count = 0
        for i in items[0: train_count]:
            train[user, i] = 1
        for i in items[train_count:]:# train_count + valid_count]:
            validation[user, i] = 1
            #for i in items[train_count + valid_count:]:
            #    test[user, i] = 1
    print("{}/{} train/valid samples".format(
        len(train.nonzero()[0]),
        len(validation.nonzero()[0])))
    return lil_matrix(train), lil_matrix(validation)

class UnitNormClipper(object):

    def __init__(self, frequency=1):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.div_(torch.norm(w, 2, 1).expand_as(w))
