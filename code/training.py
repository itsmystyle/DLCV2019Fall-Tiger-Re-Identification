import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm as tqdm

#from utils import LabelSmoothingCrossEntropy
from utils import *




def train(model, optim, dataloader, update_size=32):

	model = model.train()

	optim.zero_grad()

	batch_correct = 0.0
	batch_loss = 0.0
	total_ac = 0.0
	total_loss = 0.0

	trange = tqdm(dataloader)

	batch_count = 0.0

	for idx, data in enumerate(trange):

		if (idx+1) % update_size == 0 and idx != 0:
			optim.step()
			optim.zero_grad()
			trange.set_postfix(Accuracy=(batch_correct / batch_count), Batch_Loss=(batch_loss / batch_count))
			batch_correct = 0.0
			batch_loss = 0.0
			batch_count = 0.0


		#image, image_label, pos_image, neg_image = data
		image, erase_image, image_label = data

		image = image.cuda()
		erase_image = erase_image.cuda()
		image_label = image_label.cuda().long()


		batch_size = len(image)

		norm_feat, bn_norm_feat, erase_feat, bn_erase_feat, fuse_feat, bn_fuse_feat = model.getFeature(image, erase_image)
		
		norm_cls = model.NormalOut(bn_norm_feat)
		erase_cls = model.EraseOut(bn_erase_feat)
		fuse_cls = model.FuseOut(bn_fuse_feat)
		
		
		norm_trip = 0.0 #2.0 * TripletLoss(margin=0.4)(norm_feat, image_label)[0]
		erase_trip = 0.0 #2.0 * TripletLoss(margin=0.4)(erase_feat, image_label)[0]
		fuse_trip = 2.0 * TripletLoss(margin=0.4)(fuse_feat, image_label)[0]		
		
		norm_CE = LabelSmoothingCrossEntropy(0.1)(norm_cls, image_label)
		erase_CE = LabelSmoothingCrossEntropy(0.1)(erase_cls, image_label)
		fuse_CE = LabelSmoothingCrossEntropy(0.1)(fuse_cls, image_label)
		
		loss = norm_trip + erase_trip + fuse_trip + norm_CE + erase_CE + fuse_CE		
		loss.backward()
		'''
		triplet_loss = 2.0 * TripletLoss(margin=0.3)(image_out, image_label)[0]
		#pos_loss = nn.CosineEmbeddingLoss(margin=0.25)(image_out, model.BN_layer(pos_image_out), torch.tensor(1.0).cuda())
		#neg_loss = nn.CosineEmbeddingLoss(margin=0.25)(image_out, model.BN_layer(neg_image_out), torch.tensor(-1.0).cuda())
		cross_entropy = LabelSmoothingCrossEntropy(0.1)(image_cls, image_label)
		# cross_entropy_erase = LabelSmoothingCrossEntropy(0.1)(image_out[1], image_label)
		# cross_entropy_fuse = LabelSmoothingCrossEntropy(0.1)(image_out[2], image_label)

		loss = triplet_loss + cross_entropy #+ cross_entropy_erase + cross_entropy_fuse
		# loss = pos_loss + neg_loss + cross_entropy
		#loss = cross_entropy
		
		loss.backward()
		'''

		total_loss += loss.item()
		batch_loss += loss.item()

		pred_prob = F.softmax(fuse_cls, dim=1)
		pred = pred_prob.argmax(dim=1)
		
		correct = (pred.view(-1,1) == image_label.view(-1,1)).sum()
		batch_correct += correct.item()
		total_ac += correct.item()
		batch_count += batch_size
	
	optim.step()
	optim.zero_grad()

	total_ac /= len(dataloader.dataset)
	total_loss /= len(dataloader)

	return total_loss, total_ac



def evaluate(model, dataloader):
	
	model = model.eval()

	total_ac = 0.0
	all_pred = []

	idx2label = dataloader.dataset.gallery.id

	with torch.no_grad():

		allGaleryImg, allEraseImg = dataloader.dataset.getGalleryTensor()
		allCandidate = []
		for i in range(len(allGaleryImg)):
			img = allGaleryImg[i].cuda()
			erase_img = allEraseImg[i].cuda()
			
			fuse_feat = model.getPredFeature(img, erase_img)
			
			img = img.cpu()
			erase_img = erase_img.cpu()
			allCandidate.append(fuse_feat)

		allCandidate = torch.stack(allCandidate).squeeze(1)

		trange = tqdm(dataloader)

		for data in trange:
			
			img, erase_img, label = data
			img = img.cuda()
			erase_img = erase_img.cuda()			

			feat = model.getPredFeature(img, erase_img)

			# score = torch.mm(feat, allCandidate.transpose(0,1))
			score = F.cosine_similarity(feat.expand(allCandidate.size(0), feat.size(1)), allCandidate)
			# score = (feat.expand(allCandidate.size(0), feat.size(1)) - allCandidate).pow(2).sum(dim=1)
			score = score.view(-1,1).transpose(0,1)

			pred = score.argmax(dim=1).item()
			pred = idx2label[pred]
			correct = float(pred == label.item())
			total_ac += correct

			all_pred.append(pred.item())

	total_ac /= len(dataloader)

	return total_ac, all_pred



