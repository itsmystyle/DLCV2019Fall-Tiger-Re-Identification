import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from data import ImageDataset, QueryDataset
from train import *
from model import Model



def collate_fn(batch):
	image, self_label, pos_image, neg_image = zip(*batch)
	
	return image, self_label, pos_image, neg_image





if __name__ == '__main__':

	image_dir = "../dataset/resize_img"
	label_path = "../dataset/train.csv"
	query_path = "../dataset/query.csv"
	gallery_path = "../dataset/gallery.csv"

	train_dataset = ImageDataset(image_dir, label_path)
	train_dataloader = DataLoader(
	    train_dataset, shuffle=True, batch_size=2, num_workers=3, collate_fn=collate_fn
	)

	valid_dataset = QueryDataset(image_dir, query_path, gallery_path)
	valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=1, num_workers=3)

	model = Model(72)

	epochs = 300

	optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

	update_size = 32

	model = model.cuda()

	best_ac = -1
	out_dir = "./model/"

	for ep in range(epochs):
		print("Epoch", ep+1)
		train_loss, train_ac = train(model, optimizer, train_dataloader, update_size)
		scheduler.step(train_loss)

		valid_ac, valid_pred = evaluate(model, valid_dataloader)
		if valid_ac > best_ac:
			torch.save(model.state_dict(), out_dir + "best_model_"+str(np.round(valid_ac,5)))
			best_ac = valid_ac


		print("Training Loss:{:.5}\tTraining Accuracy:{:.5}\tValidation Accuracy:{:.5}".format(train_loss, train_ac, valid_ac))
		print("==============================")
