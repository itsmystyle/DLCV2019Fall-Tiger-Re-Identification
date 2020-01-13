import os
import glob
import numpy as np

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from module.dataset import ImageDataset
from args import argument_parser, optimizer_kwargs
import resnet
from hard_mine_triplet_loss import CenterLoss
from eval_metrics import evaluate
from of_penalty import OFPenalty
from tqdm import tqdm
from regularizer import get_regularizer
from torch.utils.data import DataLoader
import pdb

parser = argument_parser()
args = parser.parse_args()
cuda = True if torch.cuda.is_available() else False

imgs_feat = {}
id_count = {}
ground_label = []
ground_path = []
def re_ranking(
    probFea, galFea, k1, k2, lambda_value, local_distmat=None, only_local=False
):
    # if feature vector is numpy, you should use 'torch.tensor' transform it to tensor

    query_num = probFea.shape[0]
    all_num = query_num + galFea.shape[0]
    if only_local:
        original_dist = local_distmat
    else:
        feat = np.concatenate([probFea, galFea])
        feat = torch.from_numpy(feat)
        distmat = (
            torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num)
            + torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num).t()
        )
        distmat.addmm_(1, -2, feat, feat.t())
        original_dist = distmat.numpy()
        del feat
        if local_distmat is not None:
            original_dist = original_dist + local_distmat

    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, : k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, : k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[
                candidate, : int(np.around(k1 / 2)) + 1
            ]
            candidate_backward_k_neigh_index = initial_rank[
                candidate_forward_k_neigh_index, : int(np.around(k1 / 2)) + 1
            ]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(
                np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)
            ) > 2 / 3 * len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal_index
                )

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)

    original_dist = original_dist[:query_num]

    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(
                V[i, indNonZero[j]], V[indImages[j], indNonZero[j]]
            )
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]

    return final_dist


def init_optimizer(params,
                   optim='adam',
                   lr=0.003,
                   weight_decay=5e-4,
                   momentum=0.9, # momentum factor for sgd and rmsprop
                   sgd_dampening=0, # sgd's dampening for momentum
                   sgd_nesterov=False, # whether to enable sgd's Nesterov momentum
                   rmsprop_alpha=0.99, # rmsprop's smoothing constant
                   adam_beta1=0.9, # exponential decay rate for adam's first moment
                   adam_beta2=0.999 # # exponential decay rate for adam's second moment
                   ):
    if optim == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay,
                                betas=(adam_beta1, adam_beta2))
    
    elif optim == 'amsgrad':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay,
                                betas=(adam_beta1, adam_beta2), amsgrad=True)
    
    elif optim == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay,
                               dampening=sgd_dampening, nesterov=sgd_nesterov)
    
    elif optim == 'rmsprop':
        return torch.optim.RMSprop(params, lr=lr, momentum=momentum, weight_decay=weight_decay,
                                   alpha=rmsprop_alpha)
    
    else:
        raise ValueError("Unsupported optimizer: {}".format(optim))


def closet_node(node, nodes):
    nodes = torch.stack(nodes)
    if cuda:
        nodes = nodes.cuda()
    dist = torch.sum((nodes - node)**2, dim=1)
    return torch.argmin(dist)


def get_criterion(num_classes: int, use_gpu: bool, args):

    if args.criterion == 'htri':

        from hard_mine_triplet_loss import TripletLoss
        criterion = TripletLoss(num_classes, vars(args), use_gpu)

    elif args.criterion == 'xent':

        from cross_entropy_loss import CrossEntropyLoss
        criterion = CrossEntropyLoss(num_classes, use_gpu=use_gpu, label_smooth=args.label_smooth)
    else:
        raise RuntimeError('Unknown criterion {}'.format(args.criterion))

    return criterion


def pesudo_label(model, left, idx2id):
    
    model.eval()
    feat_list = list(imgs_feat.values())
    count_list = list(id_count.values())
    feat_list = [feat_list[i]/count_list[i] for i in range(len(count_list))]
    pesudo_label = []
    pesudo_path = []
    enumerator = enumerate(left)
    #pdb.set_trace()
    with torch.no_grad():
        for idx, package in enumerator:
            (imgs, pids, paths) = package
            if cuda:
                imgs = imgs.cuda()
                pids = pids.cuda()
            
            features = model(imgs, pids)[0]

            for i in range(features.shape[0]):
                pl = closet_node(features[i], feat_list).cpu().item()
                pesudo_label.append(idx2id[pl])
                pesudo_path.append(paths[i])
        
    all_label = ground_label + pesudo_label
    all_path = ground_path + pesudo_path
    with open("my_train.csv", "w") as file:
        for i in range(len(all_label)):
            file.write(str(all_label[i])+","+all_path[i]+"\n")


def test(model, query, gallery, epoch,\
    ranks=[1, 5, 10, 20], return_distmat=False):

    flip_eval = args.flip_eval

    if flip_eval:
        print('# Using Flip Eval')

    model.eval()

    with torch.no_grad():
        qf, q_pids = [], []

        if flip_eval:
            enumerator = enumerate(zip(query[0], query[1]))
        else:
            enumerator = enumerate(query)

        for batch_idx, package in enumerator:

            if flip_eval:
                (imgs0, pids), (imgs1, _) = package
                if cuda:
                    imgs0, imgs1 = imgs0.cuda(), imgs1.cuda()
                features = (model(imgs0)[0] + model(imgs1)[0]) / 2.0
                # print(features.size())
            else:
                (imgs, pids, _) = package
                if cuda:
                    imgs = imgs.cuda()
                    pids = pids.cuda()

                features = model(imgs, pids)[0]
                '''
                if epoch > 200:
                    for i in range(pids.shape[0]):
                        pid = pids[i].item()
                        if  pid not in imgs_feat:
                            imgs_feat[pid] = features[0][i].cpu().detach()
                            id_count[pid] = 1
                        else:
                            imgs_feat[pid] += features[0][i].cpu().detach()
                            id_count[pid] +=1
                '''
            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)

        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids = [], []
        if flip_eval:
            enumerator = enumerate(zip(gallery[0], gallery[1]))
        else:
            enumerator = enumerate(gallery)

        for batch_idx, package in enumerator:
            # print('fuck')

            if flip_eval:
                (imgs0, pids, camids, paths), (imgs1, _, _, _) = package
                if cuda:
                    imgs0, imgs1 = imgs0.cuda(), imgs1.cuda()
                features = (model(imgs0)[0] + model(imgs1)[0]) / 2.0
                # print(features.size())
            else:
                (imgs, pids, _) = package
                if cuda:
                    imgs = imgs.cuda()
                    pids = pids.cuda()

                features = model(imgs, pids)[0]
                '''
                if epoch > 200:
                    for i in range(pids.shape[0]):
                        pid = pids[i].item()
                        if  pid not in imgs_feat:
                            imgs_feat[pid] = features[0][i].cpu().detach()
                            id_count[pid] = 1
                        else:
                            imgs_feat[pid] += features[0][i].cpu().detach()
                            id_count[pid] +=1
                '''
            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)

        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

        if os.environ.get('save_feat'):
            import scipy.io as io
            io.savemat(os.environ.get('save_feat'), {'q': qf.data.numpy(), 'g': gf.data.numpy(), 'qt': q_pids, 'gt': g_pids})
            # return

    m, n = qf.size(0), gf.size(0)
    '''
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()
    
    '''
    distmat = re_ranking(
            qf, gf, k1=25, k2=6, lambda_value=0.3
        )
    
    preds = distmat.argmin(axis=1)

    '''
    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids,\
         use_metric_cuhk03=args.use_metric_cuhk03)

    print("Results ----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r - 1]))
    print("------------------")
    '''
    if return_distmat:
        return distmat
    return sum(preds==q_pids)/len(q_pids)


dataset = ImageDataset("imgs", "train.csv", train=True)
train_dataloader = DataLoader(
    dataset, shuffle=True, batch_size=8, num_workers=32
)
query = ImageDataset("imgs", "query.csv", train=False)
gallery = ImageDataset("imgs", "gallery.csv", train=False)
query_dataloader = DataLoader(
    query, shuffle=False, batch_size=8, num_workers=32
)
gallery_dataloader = DataLoader(
    gallery, shuffle=False, batch_size=8, num_workers=32
)
left = ImageDataset("imgs", "left.csv", train=False)
left_dataloader = DataLoader(
    left, shuffle=False, batch_size=8, num_workers=32
)
criterion = get_criterion(len(dataset.id2idx), cuda, args)
model = resnet.resnet50(len(dataset.id2idx), vars(args))
regularizer = get_regularizer(vars(args))
optimizer = init_optimizer(model.parameters(), **optimizer_kwargs(args))
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.stepsize,\
            gamma=args.gamma)
if cuda:
    model = nn.DataParallel(model).cuda()

device = torch.device("cuda" if cuda else "cpu")
center_loss = CenterLoss(len(dataset.id2idx), 3072, device=device)
_best = 0
for epoch in range(350):

    model.train()
    '''
    dataset = ImageDataset("imgs", "my_train.csv", train=True)
    train_dataloader = DataLoader(
        dataset, shuffle=True, batch_size=8, num_workers=32
    )
    '''
    all_loss = 0.0
    trange = tqdm(
        enumerate(train_dataloader),
        total = len(train_dataloader),
        desc = "Epoch {}".format(epoch)
    )
    n_total = 0
    n_correct = 0

    for idx, (imgs, label, path) in trange:
        
        if epoch < 1:
            for i in range(label.shape[0]):
                ground_label.append(dataset.idx2id[label[i].item()])
                ground_path.append(path[i])

        if cuda:
            imgs = imgs.cuda()
            label = label.cuda()

        of_penalty = OFPenalty(vars(args))
        output = model(imgs, label)
        loss = criterion(output, label)
        reg = regularizer(model)
        loss += reg
        center = center_loss(output[0], label)
        loss += 0.001 * center
        penalty = of_penalty(output)
        loss += penalty
        #pdb.set_trace()
        '''
        if epoch > 200:
            ids = label.cpu().detach()
            for i in range(ids.shape[0]):
                id = ids[i].item()
                if  id not in imgs_feat:
                    imgs_feat[id] = output[0][i].cpu().detach()
                    id_count[id] = 1
                else:
                    imgs_feat[id] += output[0][i].cpu().detach()
                    id_count[id] += 1
        '''
        #optimizer.zero_grad()
        loss.backward()

        #optimizer.step()
        
        if (idx+1)%4==0:
            optimizer.step()
            optimizer.zero_grad()
        
        all_loss += loss.cpu().detach().item()    
        trange.set_postfix(
                {'loss':"{0:.3f}".format(all_loss / (idx + 1))}
            )

    scheduler.step()

    if epoch % 1 == 0:
        rank1 = test(model, query_dataloader, gallery_dataloader, epoch)
        print(rank1)
    
    if _best < rank1:
        _best = rank1
        torch.save(model.state_dict(), os.path.join("./ctal_2", 'model_{}.pth.tar'.format(epoch)))
    '''
    if epoch > 200:
        pesudo_label(model, left_dataloader, dataset.idx2id)
    '''