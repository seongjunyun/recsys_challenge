import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import pickle
from torch.autograd import grad
from torch.autograd import Variable
from model import CML
from copy import deepcopy
from evaluator_pytorch import RecallEvaluator
from sampler import WarpSampler
from ranking import TopK
from tqdm import tqdm
from scipy.sparse import lil_matrix
import toolz
import functools
import gc

class Solver(object):

    def __init__(self, train_data, valid_data, features, artists_dic, albums_dic, titles, titles_len, ratios, val_candidates, unigram_probs, config):
        
        #Data loader
        self.train_data = train_data
        self.val_candidates = val_candidates
        self.valid_data = valid_data
        self.features = features
        self.artists_dic = artists_dic
        self.albums_dic = albums_dic
        self.titles = titles
        self.titles_len = titles_len 
        self.ratios = ratios 
        self.unigram_probs = unigram_probs
        # Model Configuration
        self.n_users = config.n_users
        self.n_items = config.n_items
        self.n_negative = config.n_negative
        self.embed_dim = config.embed_dim
        self.n_artist = config.n_artist
        self.n_album = config.n_album
        self.n_title = config.n_title
        # Training Configuration
        self.mode = config.mode
        self.lr = config.lr   
        self.clip_norm = config.clip_norm     
        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs
        self.optim_size = config.optim_size 
        # Validation settings
        self.best_loss = 100000

        # Step size
        self.log_step = config.log_step
        self.model_save_step = config.model_save_step
        total_size = 64301450 #64670417#65367813#66627428#64644977#
        self.evaluation_every_n_batchs = int(total_size/self.batch_size) + 1

        # Directories
        self.log_dir = config.log_dir
        self.model_save_dir = config.model_save_dir
 
        # log
        self.stats = {}

        # Build model
        self.build_model()
        print(self.optim_size / self.batch_size)
    def build_model(self):
        # Define CML
        self.model = CML(self.n_users,
                         self.n_items,
                         self.n_artist,
                         self.n_album,
                         self.n_title,
                         features=None,
                         embed_dim=self.embed_dim,
                         margin=2.0,
                         use_rank_weight=True,
                         use_cov_loss=False,
                         cov_loss_weight=1
                         )
        
        if torch.cuda.is_available():
            self.model.cuda()# = nn.DataParallel(self.model).cuda()

        # Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)#, weight_decay=1e-6)
        
        # Print networks
        self.print_network(self.model, 'CML')
    
    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))


    def restore_model(self, resume_iters):
        """Restore the CML model."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        CML_path = os.path.join(self.model_save_dir, '{}-CML.ckpt'.format(resume_iters))
        self.model.load_state_dict(torch.load(CML_path, map_location=lambda storage, loc: storage))
    
    def load_model(self):
        """Load the best CML model."""
        print('Loading the best CML model')
        CML_path = os.path.join(self.model_save_dir, 'best-CML_8.ckpt')
        self.model.load_state_dict(torch.load(CML_path, map_location=lambda storage, loc: storage))

    def save_model(self):
        # Save model checkpoints
        torch.save(self.model.state_dict(), os.path.join(self.model_save_dir, 'best-CML_6.ckpt'))


    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def to_var(self, x, volatile=True):
        if torch.cuda.is_available():
            x = x.cuda()
        with torch.no_grad():
            x = Variable(x)
        return x
    def from_numpy(self, x):
        x = self.to_var(torch.from_numpy(x))

        return x

    def clip_by_norm(self, parameter):
        l2norm = torch.norm(parameter.weight.data, 2, 1)
        is_clipping = l2norm > 2.3
        if len(l2norm[is_clipping]) > 0:
            parameter.weight.data[is_clipping] = parameter.weight.data[is_clipping] * 2.3 / l2norm[is_clipping].unsqueeze(1) 
   
    def get_loss(self, pos_distances, distance_to_neg_items, closest_negative_item_distances, margin=2.0):

        # compute hinge loss (N)
        loss_per_pair = torch.relu(pos_distances - closest_negative_item_distances + margin)

        #if self.use_rank_weight:
        # indicator matrix for impostors (N x W)
        impostors = (torch.unsqueeze(pos_distances, -1) - distance_to_neg_items + margin) > 0
        # approximate the rank of positive item by (number of impostor / W per user-positive pair)
        rank = torch.mean(impostors.type(torch.float32), 1) * self.n_items
        # apply rank weight
        loss_per_pair *= torch.log(rank + 1)

        # the embedding loss
        #total_distance = torch.cat((pos_distances.unsqueeze(-1), distance_to_neg_items), 1)
        #target = torch.zeros(total_distance.shape[0], dtype=torch.long).cuda()

        #loss = self.loss(total_distance, target)     
        loss = torch.sum(loss_per_pair)

        return loss
   
    
    def train(self):
      
        print('train start') 
        #train, valid, artists_dic, albums_dic, titles, titles_len = spotify(self.mode, self.valid) 
        sampler = WarpSampler(self.train_data, self.unigram_probs, batch_size=self.batch_size, n_negative=self.n_negative, check_negative=True)

        # sample some users to calculate recall validation
        items = self.from_numpy(np.arange(self.n_items)) 
        prev_recall = 0
        recall_score = 0.0000001
        prev_ndcg = 0
        ndcg_score = 0.0000001
        epoch = 1
        while epoch <= self.num_epochs :

            self.model.train()
            if prev_recall < recall_score and prev_ndcg < ndcg_score:
                
                prev_recall = recall_score
                prev_ndcg = ndcg_score
            self.save_model()
            print('Model saved')

            # train model
            losses = []
            # run n mini-batches
            
            #try:
            #    for obj in gc.get_objects():
            #        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            #            print(type(obj), obj.size()) 
            #except:
            #    pass
            #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            for i in tqdm(range(self.evaluation_every_n_batchs), desc="Optimizing..."):
                
                user_pos, neg = sampler.next_batch()
                pos_artists = self.from_numpy(self.artists_dic[user_pos[:, 1]])
                pos_albums = self.from_numpy(self.albums_dic[user_pos[:, 1]])
                neg_artists = self.from_numpy(np.array([self.artists_dic[negative_sample] for negative_sample in neg])).type(torch.long)
                neg_albums = self.from_numpy(np.array([self.albums_dic[negative_sample] for negative_sample in neg])).type(torch.long)
                titles = None#self.from_numpy(self.titles[user_pos[:, 0]])
                titles_len = None#self.from_numpy(self.titles_len[user_pos[:, 0]])  
                user_pos = self.from_numpy(user_pos).type(torch.long)
                neg = self.from_numpy(neg).type(torch.long)
                self.model.zero_grad()                 
                pos_distances, distance_to_neg_items, closest_negative_item_distances = self.model(user_pos, pos_artists, pos_albums, neg, neg_artists, neg_albums, titles, titles_len)# / (self.optim_size / self.batch_size)
                loss = self.get_loss(pos_distances, distance_to_neg_items, closest_negative_item_distances)
                loss.backward(retain_graph=False)
                #torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip_norm)
                self.optimizer.step()
                self.clip_by_norm(self.model.module.user_embeddings)
                self.clip_by_norm(self.model.module.item_embeddings)
                self.clip_by_norm(self.model.module.artist_embeddings)
                self.clip_by_norm(self.model.module.album_embeddings)
                #self.clip_by_norm(self.model.title_embeddings)
                #self.model.title_embeddings.weight.data = self.clip_by_norm(self.model.title_embeddings.weight.data)
                #if (i+1) % (self.optim_size / self.batch_size) == 0 or i == self.evaluation_every_n_batchs-1:
                #    self.optimizer.step() 
                #    self.model.zero_grad()
                
                losses.append(loss.detach().cpu().numpy())
            
            torch.cuda.empty_cache()
            print("\nTraining loss {}".format(np.mean(losses)))
            epoch += 1

             # compute recall in chunks to utilize speedup provided by Tensorflow
            artists = self.from_numpy(self.artists_dic)
            albums = self.from_numpy(self.albums_dic) 
            titles = self.from_numpy(self.titles) 
            titles_len = self.from_numpy(self.titles_len) 
            #ratios = self.from_numpy(self.ratios) 
            self.model.eval() 
            
            for i in range(10):
                # create evaluator on validation set
                validation_recall = RecallEvaluator(self.model, self.train_data, self.valid_data[i])
                # compute recall on validate set
                valid_recalls = []
                valid_ndcgs = []
                valid_users = np.array(list(set(self.valid_data[i].nonzero()[0])), )
                #valid_users = list(set(self.valid_data[i].nonzero()[0]))
                for user_chunk in toolz.partition_all(50, valid_users):
                    user_chunk = self.from_numpy(np.array(user_chunk)).type(torch.long)
                    recall, ndcg = validation_recall.eval(user_chunk, items, artists, albums, titles[user_chunk], titles_len[user_chunk], None)
                    valid_recalls.extend(recall)
                    valid_ndcgs.extend(ndcg)

                
                #for user_chunk in valid_users:
                    #items = np.array(self.val_candidates[user_chunk])
                    #if len(items) > 50:
                        #artists = self.from_numpy(self.artists_dic[items]).type(torch.long)
                        #albums = self.from_numpy(self.albums_dic[items]).type(torch.long)
                        #items = self.from_numpy(items).type(torch.long)
                        #user_chunk = self.from_numpy(np.array([user_chunk])).type(torch.long)
                        #recall, ndcg = validation_recall.eval(user_chunk, items, artists, albums, titles[user_chunk], titles_len[user_chunk], ratios[user_chunk-990000])
                        #valid_recalls.extend([recall])
                        #valid_ndcgs.extend([ndcg])
                    #else:
                        #print(len(items))
                        #valid_recalls.extend([[0.0]])
                        #valid_ndcgs.extend([[0.0]])
                
                 
                recall_score = np.mean(valid_recalls)
                ndcg_score = np.mean(valid_ndcgs) 
                print('\nNo. {}'.format(i))
                print("Recall on (sampled) validation set: {}".format(recall_score))
                print("Ndcg on (sampled) validation set: {}".format(ndcg_score))
            print('Epoch: {}'.format(epoch))
        torch.cuda.empty_cache()
        self.predict()

    def predict(self):
        torch.cuda.empty_cache()
        self.load_model()         
        valid_users = np.arange(1000000, 1010000)
        Ranking = TopK(self.model, self.train_data)
        items = self.from_numpy(np.arange(self.n_items)) 
        prev_recall = 0
        epoch = 1
        artists = self.from_numpy(self.artists_dic)
        albums = self.from_numpy(self.albums_dic) 
        titles = self.from_numpy(self.titles) 
        titles_len = self.from_numpy(self.titles_len) 
         
        results = []
        scores = []
        for user_chunk in toolz.partition_all(200, valid_users):
            user_chunk = self.from_numpy(np.array(user_chunk)).type(torch.long)
            result, score = Ranking.predict(user_chunk, items, artists, albums)
            results.extend(result)#, titles[user_chunk], titles_len[user_chunk])
            scores.extend(score)
        with open('results/'+'cml_results_9.pickle','wb') as f:
            pickle.dump(results,f)
        with open('results/'+'cml_scores_9.pickle','wb') as f:
            pickle.dump(scores,f)


    def valid(self):
        self.load_model()         
        #valid_users = np.array(list(set(self.valid_data.nonzero()[0])), )
        items = self.from_numpy(np.arange(self.n_items)) 
        artists = self.from_numpy(self.artists_dic)
        albums = self.from_numpy(self.albums_dic) 
        titles = self.from_numpy(self.titles) 
        titles_len = self.from_numpy(self.titles_len) 
         
        for i in range(10):
                # create evaluator on validation set
                validation_recall = RecallEvaluator(self.model, self.train_data, self.valid_data[i])
                # compute recall on validate set
                valid_recalls = []
                valid_ndcgs = []
                valid_users = np.array(list(set(self.valid_data[i].nonzero()[0])), )
                #valid_users = list(set(self.valid_data[i].nonzero()[0]))
                for user_chunk in toolz.partition_all(400, valid_users):
                    user_chunk = self.from_numpy(np.array(user_chunk)).type(torch.long)
                    recall, ndcg = validation_recall.eval(user_chunk, items, artists, albums, titles[user_chunk], titles_len[user_chunk], None)
                    valid_recalls.extend(recall)
                    valid_ndcgs.extend(ndcg)      
                 
                recall_score = np.mean(valid_recalls)
                ndcg_score = np.mean(valid_ndcgs) 
                print('\nNo. {}'.format(i))
                print("Recall on (sampled) validation set: {}".format(recall_score))
                print("Ndcg on (sampled) validation set: {}".format(ndcg_score))

