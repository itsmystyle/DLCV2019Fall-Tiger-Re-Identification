import os
import argparse

import pandas as pd

from sklearn.metrics import accuracy_score

def read_csv(csv_path):
    return pd.read_csv(csv_path, header=None, index_col=False)

def evaluate(query, gallery, pred):
    
    assert query.shape[0] == pred.shape[0]
    
    pred = pred.squeeze()
    qurey_id = query[:, 0].tolist()
    pred_id  = []
    gallery_dic = dict(zip(gallery[:,1], gallery[:,0]))
    
    for p in pred:
        pred_id.append(gallery_dic[p])
    
    return accuracy_score(qurey_id, pred_id)*100    


if __name__ == '__main__':

    '''argument parser'''
    parser = argparse.ArgumentParser(description='Code to evaluate DLCV final challenge 1')
    parser.add_argument('--query', type=str, help='path to query.csv') 
    parser.add_argument('--gallery', type=str, help='path to gallery.csv')
    parser.add_argument('--pred', type=str, help='path to your predicted csv file (e.g. predict.csv)')
    args = parser.parse_args()

    ''' read csv files '''
    query = read_csv(args.query)
    gallery = read_csv(args.gallery)
    pred = read_csv(args.pred)
    

    rank1 = evaluate(query.values, gallery.values, pred.values)
    
    print('===> rank1: {}%'.format(rank1))

