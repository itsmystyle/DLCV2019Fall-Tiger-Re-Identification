import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import math

import torch.hub
import pretrainedmodels


class Model(nn.Module):


	def __init__(self, class_num):
		super(Model, self).__init__()

		self.class_num = class_num

		self.backbone = torch.hub.load('moskomule/senet.pytorch', 'se_resnet50', pretrained=True)
		
		# Modify the last conv stride to 1
		#self.backbone = pretrainedmodels.se_resnext101_32x4d(num_classes=1000, pretrained='imagenet')
		self.backbone.layer4[0].downsample[0].stride = (1, 1)
		self.backbone.layer4[0].conv2.stride = (1, 1)

		self.getFeat = nn.Sequential(
			self.backbone.conv1,
			self.backbone.bn1,
			self.backbone.relu,
			self.backbone.maxpool,
			
			#self.backbone.layer0,
			self.backbone.layer1,
			self.backbone.layer2,
			self.backbone.layer3,
			self.backbone.layer4,
			#self.backbone.avgpool,
		)

		self.spatial_pyramid_pooling = SpatialPyramidPool([1], mode='avg')
		#self.spatial_pyramid_pooling = nn.AdaptiveAvgPool2d(1)

		self.BN_layer = nn.BatchNorm1d(2048)
		# self.BN_layer.bias.requires_grad_(False)
		self.NormalOut = nn.Linear(2048, class_num, bias=False)

		# initialization:
		self.BN_layer.apply(weights_init_kaiming)
		self.NormalOut.apply(weights_init_classifier)

	def forward(self, img):

		pool_img_feat = self.getFeature(img)

		return pool_img_feat



	def getFeature(self, img):

		img_feat = self.getFeat(img)
		pool_img_feat = self.spatial_pyramid_pooling(img_feat).flatten(start_dim=1)
		#pool_img_feat = img_feat.flatten(start_dim=1)

		return pool_img_feat


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)



def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)



# https://github.com/revidee/pytorch-pyramid-pooling
# https://arxiv.org/pdf/1406.4729.pdf
class SpatialPyramidPool(nn.Module):

	def __init__(self, levels, mode='max'):
		super(SpatialPyramidPool, self).__init__()

		self.levels = levels
		self.mode = mode

	def forward(self, inp_feat):

		batch_size = inp_feat.size(0)

		for i in range(len(self.levels)):

			h, w = inp_feat.shape[2:]

			ks = (h // self.levels[i], w // self.levels[i])

			if self.mode == 'max':
				pool = nn.MaxPool2d(kernel_size=ks, stride=ks)
			if self.mode == 'avg':
				pool = nn.AvgPool2d(kernel_size=ks, stride=ks)

			spp = pool(inp_feat)

			if i == 0:
				out = spp.view(batch_size, -1)
			else:
				out = torch.cat([out, spp.view(batch_size, -1)], dim=1)

		return out




# if __name__ == '__main__':
# 	model = Model(72)
# 	inp = torch.randn(1,3,128,128)

