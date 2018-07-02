import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class CML(nn.Module):

    def __init__(self,
                 n_users,
                 n_items,
                 n_artists,
                 n_albums,
                 n_titles,
                 embed_dim=100,
                 features=None,
                 margin=1.5,
                 #master_learning_rate=0.01,
                 #clip_norm=1.0,
                 hidden_layer_dim=128,
                 dropout_rate=0.2,
                 feature_l2_reg=0.1,
                 feature_projection_scaling_factor=0.5,
                 use_rank_weight=True,
                 use_cov_loss=True,
                 cov_loss_weight=0.1
                 ):
        super(CML, self).__init__()
        
        #non feature
        self.n_users = n_users
        self.n_items = n_items
        self.n_artists = n_artists
        self.n_albums = n_albums
        self.n_titles = n_titles
        self.embed_dim = embed_dim

        self.margin = margin
        self.use_rank_weight = use_rank_weight
        self.use_cov_loss = use_cov_loss
        self.cov_loss_weight = cov_loss_weight
        
        self.user_embeddings = nn.Embedding(self.n_users, self.embed_dim)
        self.item_embeddings = nn.Embedding(self.n_items, self.embed_dim)
        self.artist_embeddings = nn.Embedding(self.n_artists, self.embed_dim)
        self.album_embeddings = nn.Embedding(self.n_albums, self.embed_dim)
        #self.title_embeddings = nn.Embedding(self.n_titles, self.embed_dim, padding_idx=2464)

        #self.sq_tan_alpha = 0.527864#1#1.6197752
        
        #self.loss = nn.CrossEntropyLoss()
        
        #self.layer = nn.Sequential(nn.Linear(300,150),
        #                           nn.ReLU(),
        #                           nn.Dropout(0.3),
        #                           nn.Linear(150,100))
      
        #feature  
        #if features is not None:
        #    self.features = tf.constant(features, dtype=tf.float32)
        #else:
        #    self.features = None

        #self.hidden_layer_dim = hidden_layer_dim
        #self.dropout_rate = dropout_rate
        #self.feature_l2_reg = feature_l2_reg
        #self.feature_projection_scaling_factor = feature_projection_scaling_factor

        self.init_weight() 
    
    def init_weight(self):
        
        nn.init.normal_(self.user_embeddings.weight, std=1 / self.embed_dim ** 0.5)
        nn.init.normal_(self.item_embeddings.weight, std=1 / self.embed_dim ** 0.5)
        nn.init.normal_(self.album_embeddings.weight, std=1 / self.embed_dim ** 0.5)
        nn.init.normal_(self.artist_embeddings.weight, std=1 / self.embed_dim ** 0.5)
        #nn.init.normal_(self.title_embeddings.weight, std=1 / self.embed_dim ** 0.5)
        #self.title_embeddings.weight.data[2464] = 0
     
    
    def forward(self, user_positive_items_pairs,  pos_artists, pos_albums, neg_samples, neg_artists, neg_albums, titles, titles_len):
        """
        :return: the distance metric loss
        """
        # Let
        # N = batch size,
        # K = embedding size,
        # W = number of negative samples per a user-positive-item pair
 
        # user embedding (N, K)
        users = self.user_embeddings(user_positive_items_pairs[:, 0])
        # user title embedding (N, ?)
        #user_titles = torch.sum(self.title_embeddings(titles), 1)
        # user title len (N)
        #titles_len = titles_len.unsqueeze(1).type(torch.float32)
        #user_titles = user_titles / titles_len
        #print(user_titles)
        #users = (users + user_titles)/2 
        # positive item embedding (N, K)
        #pos_items = self.item_embeddings(user_positive_items_pairs[:, 1])
        # positive artist embedding (N, K)
        #pos_artists = self.artist_embeddings(pos_artists)
        # positive album embedding (N, K)
        #pos_albums = self.album_embeddings(pos_albums)
        # positive : (1 + w_artist + w_album) * v_song
       
        #pos_items = self.layer(torch.cat((pos_items, pos_artists, pos_albums),1))
        pos_items = (self.item_embeddings(user_positive_items_pairs[:, 1]) + self.artist_embeddings(pos_artists) + self.album_embeddings(pos_albums) )/ 3

        # positive item to user distance (N)
        #pos_distances = torch.norm(users - pos_items,2, 1)
        pos_distances = torch.sum((users - pos_items)**2, 1)
        #pos_distances = torch.sum(users * pos_items, 1)
        # negative item embedding (N, K, W)
        
        #neg_items = self.item_embeddings(neg_samples).permute(0,2,1)
        
        # negative artist embedding (N, K)
        #neg_artists = self.artist_embeddings(neg_artists).permute(0,2,1)
        # negative album embedding (N, K)
        #neg_albums = self.album_embeddings(neg_albums).permute(0,2,1)
        # negative : (1 + w_artist + w_album) * v_song
        #neg_items = self.layer(torch.cat((neg_items, neg_artists, neg_albums),2)).permute(0,2,1)
        #neg_items = (neg_items + neg_artists + neg_albums) / 3
        neg_items = (self.item_embeddings(neg_samples).permute(0,2,1) + self.artist_embeddings(neg_artists).permute(0,2,1) + self.album_embeddings(neg_albums).permute(0,2,1)) / 3
        
        # angular loss : x_c
        #c = (users + pos_items) / 2
        
        # distance to negative items (N x W)
        #distance_to_neg_items = torch.norm(torch.unsqueeze(users, -1)-neg_items,2, 1)
        distance_to_neg_items = torch.sum((torch.unsqueeze(users, -1)-neg_items)**2, 1)
        #distance_to_neg_items = torch.sum(torch.unsqueeze(users, -1) * neg_items, 1)
        

        # best negative item (among W negative samples) their distance to the user embedding (N)
        closest_negative_item_distances, _ = torch.min(distance_to_neg_items, 1)

        # compute hinge loss (N)
        #loss_per_pair = torch.relu(pos_distances - closest_negative_item_distances + self.margin)

        #if self.use_rank_weight:
            # indicator matrix for impostors (N x W)
        #    impostors = (torch.unsqueeze(pos_distances, -1) - distance_to_neg_items + self.margin) > 0
            # approximate the rank of positive item by (number of impostor / W per user-positive pair)
        #    rank = torch.mean(impostors.type(torch.float32), 1) * self.n_items
            # apply rank weight
        #    loss_per_pair *= torch.log(rank + 1)

        # the embedding loss
        #total_distance = torch.cat((pos_distances.unsqueeze(-1), distance_to_neg_items), 1)
        #target = torch.zeros(total_distance.shape[0], dtype=torch.long).cuda()

        #loss = self.loss(total_distance, target)     
        #loss = torch.sum(loss_per_pair)
        
        return pos_distances, distance_to_neg_items, closest_negative_item_distances


    def item_scores2(self, user_ids, item_ids, artist_ids, album_ids , a, b, c):
        
        with torch.no_grad():
            # (N_USER_IDS, 1, K)
            user = self.user_embeddings(user_ids).unsqueeze(1).detach()
            # (1, N_ITEM, K)
            item = self.item_embeddings(item_ids).unsqueeze(0).detach()
            artist = self.artist_embeddings(artist_ids).unsqueeze(0).detach()
            album = self.album_embeddings(album_ids).unsqueeze(0).detach()        
            #item = self.layer(torch.cat((item, artist, album), 2))
            item = (item + artist + album ) #/ 3
            # score = minus distance (N_USER, N_ITEM) 
            #score = -torch.sum((user-item)** 2, 2) 
            #score = -torch.norm(user-item, 2, 2) 
            score = torch.sum(user*item, 2) 
            total_score = score
        
        return total_score




    def item_scores(self, user_ids, item_ids, artist_ids, album_ids, title_ids, title_len, ratios):
        
        step = 10000
        # (N_USER_IDS, 1, K)
        user = self.user_embeddings(user_ids).unsqueeze(1).detach()
        #title = torch.sum(self.title_embeddings(title_ids).detach(), 1)
        #title_len = title_len.unsqueeze(1).type(torch.float32)
        #title = (title / title_len).unsqueeze(1)
        #user = (user + title) / 2
        #ratios_song = ratios[:,0].unsqueeze(1).unsqueeze(1).detach()
        #ratios_artist = ratios[:,1].unsqueeze(1).unsqueeze(1).detach()
        #ratios_album = ratios[:,2].unsqueeze(1).unsqueeze(1).detach()
        for i in range(0, len(item_ids), step):
            # (1, N_ITEM, K)
            item = self.item_embeddings(item_ids[i: i+step]).unsqueeze(0).detach()
            artist = self.artist_embeddings(artist_ids[i: i+step]).unsqueeze(0).detach()
            album = self.album_embeddings(album_ids[i: i+step]).unsqueeze(0).detach()        
            #item = self.layer(torch.cat((item, artist, album), 2))
            item = (item + 1.5*artist + album ) / 3.5
            #item = (ratios_song * item + ratios_artist * artist + ratios_album * album ) / (ratios_song + ratios_artist + ratios_album)
            # score = minus distance (N_USER, N_ITEM) 
            #score = -torch.norm(user-item, 2, 2) 
            score = -torch.sum((user-item)**2, 2) 
            #score = torch.sum(user*item, 2)  

            if i ==0:
                total_score = score
            else:
                total_score = torch.cat((total_score, score), 1)
        
        return total_score

def test():
    
    model = CML(100, 100, 100, 100, 2465)
    pair = Variable(torch.LongTensor(100,2).random_(0,20))
    artist = Variable(torch.LongTensor(100).random_(0,20))
    album = Variable(torch.LongTensor(100).random_(0,20))
    neg_sample = Variable(torch.LongTensor(100,50).random_(0,20))
    neg_artist = Variable(torch.LongTensor(100,50).random_(0,20))
    neg_album = Variable(torch.LongTensor(100,50).random_(0,20))
    titles = Variable(torch.LongTensor(100,8).random_(0,20))
    titles_len = Variable(torch.LongTensor(100).random_(0,6))

    a = model(pair, artist, album, neg_sample, neg_artist, neg_album, titles, titles_len)
    
    user_ids = Variable(torch.LongTensor(20).random_(0,20))
    item_ids = Variable(torch.LongTensor(1000).random_(0,20))
    artist_ids = Variable(torch.LongTensor(1000).random_(0,20))
    album_ids = Variable(torch.LongTensor(1000).random_(0,20))
    title_ids = Variable(torch.LongTensor(20, 8).random_(0,20))
    titles_len = Variable(torch.LongTensor(20).random_(0,7))

    score = model.item_scores(user_ids, item_ids, artist_ids, album_ids, title_ids, titles_len)

#test()

