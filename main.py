import os
import argparse
from solver import Solver
import numpy as np
from utils import spotify
from scipy.sparse import lil_matrix

def main(config):

    # Create directories if not exist
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)

    train, valid, features, artists_dic, albums_dic, titles, titles_len, ratios, val_candidates, unigram_probs = spotify(config.mode, 'False') 
    # Solver
    solver = Solver(train, valid, features, artists_dic, albums_dic, titles, titles_len, ratios, val_candidates , unigram_probs, config)

    if config.mode == 'train':
        print('train')
        solver.train()
    elif config.mode == 'test':
        print('test')
        solver.predict()
    elif config.mode == 'valid':
        print('valid')
        solver.valid()

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    #Model hyper-parameters
    parser.add_argument('--n_users', type=int, default=1010000)
    parser.add_argument('--n_items', type=int, default=1189252)#2262292)#1177357)
    parser.add_argument('--n_artist', type=int, default=150282)#295860)#148134)
    parser.add_argument('--n_album', type=int, default=402213)#734684)#397927)
    parser.add_argument('--n_title', type=int, default=2465)
    parser.add_argument('--embed_dim', type=int, default=130)
    parser.add_argument('--n_negative', type=int, default=100)
    # Training settings
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--clip_norm', type=float, default=2.0)
    parser.add_argument('--batch_size', type=int, default=45000)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--optim_size', type=int, default=60000)

    # Directories
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--model_save_dir', type=str, default='./models')

    # Step size
    parser.add_argument('--log_step', type=int, default=50)
    parser.add_argument('--model_save_step', type=int, default=50000)
    
    # main
    parser.add_argument('--mode', type=str, default='train', choices=['train','test', 'valid'])
    config = parser.parse_args()
    print(config)
    main(config)
