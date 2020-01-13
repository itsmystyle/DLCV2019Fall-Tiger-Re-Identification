#import matplotlib.pyplot as plt
import csv
from collections import defaultdict
import glob
import pandas as pd
import pdb

_dict = defaultdict(int)
with open('train.csv', newline='') as csvfile:
    rows = csv.reader(csvfile)

    for row in rows:
        _dict[int(row[0])] +=1

print(len(_dict))
print(_dict)

_dict = defaultdict(int)
with open('query.csv', newline='') as csvfile:
    rows = csv.reader(csvfile)

    for row in rows:
        _dict[int(row[0])] +=1

print(len(_dict))
print(_dict)

_dict = defaultdict(int)
with open('gallery.csv', newline='') as csvfile:
    rows = csv.reader(csvfile)

    for row in rows:
        _dict[int(row[0])] +=1

print(len(_dict))
print(_dict)

'''
train = pd.read_csv("train.csv", header=None, names=["id", "img_file"])
query = pd.read_csv("query.csv", header=None, names=["id", "img_file"])
gallery = pd.read_csv("gallery.csv", header=None, names=["id", "img_file"])

train_list = train["img_file"].values.tolist()
query_list = query["img_file"].values.tolist()
gallery_list = gallery["img_file"].values.tolist()

current = train_list + query_list + gallery_list
imgs_path = glob.glob("./imgs/*jpg")
all = [ip.split("/")[-1] for ip in imgs_path]

left = list(set(all) - set(current))
pdb.set_trace()

with open('left.csv', 'w', newline='')as csvfile:
    
    writer = csv.writer(csvfile)
    for img in left:
        writer.writerow([0, img])
'''