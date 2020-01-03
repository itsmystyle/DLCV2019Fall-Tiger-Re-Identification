import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm as tqdm

from utils import LabelSmoothingCrossEntropy

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

		if (idx+1) % update_size == 0:
			optim.step()
			optim.zero_grad()
			trange.set_postfix(Accuracy=(batch_correct / batch_count), Batch_Loss=(batch_loss / batch_count))
			batch_correct = 0.0
			batch_loss = 0.0
			batch_count = 0.0


		image, image_label, pos_image, neg_image = data

		image_out = []
		pos_image_out = []
		neg_image_out = []

		batch_size = len(image)

		for i in range(batch_size):
			image_out.append( model.getFeature(image[i].cuda().unsqueeze(0)))
			pos_image_out.append( model.getFeature(pos_image[i].cuda().unsqueeze(0)))
			neg_image_out.append( model.getFeature(neg_image[i].cuda().unsqueeze(0)))

		image_out = torch.stack(image_out).squeeze(1)
		pos_image_out = torch.stack(pos_image_out).squeeze(1)
		neg_image_out = torch.stack(neg_image_out).squeeze(1)

		image_cls = model.NormalOut(model.BN_layer(image_out))
		image_label = torch.Tensor(image_label).cuda().long()

		# image_out = model(image)
		# pos_image_out = model(pos_image)[-1].detach()
		# neg_image_out = model(neg_image)[-1].detach()

		triplet_loss = nn.TripletMarginLoss(margin=0.3, p=2)(image_out, pos_image_out, neg_image_out)
		# pos_loss = nn.CosineEmbeddingLoss(margin=0.25)(image_out[1], pos_image_out, torch.tensor(1.0).cuda())
		# neg_loss = nn.CosineEmbeddingLoss(margin=0.25)(image_out[1], neg_image_out, torch.tensor(-1.0).cuda())
		cross_entropy = LabelSmoothingCrossEntropy(0.1)(image_cls, image_label)
		# cross_entropy_erase = LabelSmoothingCrossEntropy(0.1)(image_out[1], image_label)
		# cross_entropy_fuse = LabelSmoothingCrossEntropy(0.1)(image_out[2], image_label)

		loss = triplet_loss + cross_entropy #+ cross_entropy_erase + cross_entropy_fuse
		# loss = pos_loss + neg_loss + cross_entropy
		# loss = cross_entropy
		loss.backward()

		total_loss += loss.item()
		batch_loss += loss.item()

		pred_prob = F.softmax(image_cls, dim=1)
		pred = pred_prob.argmax(dim=1)
		
		correct = (pred.view(-1,1) == image_label.view(-1,1)).sum()
		batch_correct += correct.item()
		total_ac += correct.item()
		batch_count += batch_size


	total_ac /= len(dataloader.dataset)
	total_loss /= len(dataloader)

	return total_loss, total_ac



def evaluate(model, dataloader):
	
	model = model.eval()

	total_ac = 0.0
	all_pred = []

	idx2label = dataloader.dataset.gallery.id

	with torch.no_grad():

		allGaleryImg = dataloader.dataset.getGalleryTensor()
		allCandidate = []
		for i in range(len(allGaleryImg)):
			img = allGaleryImg[i].cuda()
			feat = model.BN_layer(model.getFeature(img))
			# feat = model.getFeature(img)[-2]
			img = img.cpu()
			allCandidate.append(feat)
		allCandidate = torch.stack(allCandidate).squeeze(1)


		trange = tqdm(dataloader)

		for data in trange:
			
			img, label = data
			img = img.cuda()

			feat = model.BN_layer(model.getFeature(img))
			# feat = model.getFeature(img)[-2]

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
