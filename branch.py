import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from torchtools import init_params
from shallow_cam import get_attention_module_instance
from module.utils import ArcMarginProduct
import pdb


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiBranchNetwork(nn.Module):

    def __init__(self, backbone, args, num_classes, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.common_branch = self._get_common_branch(backbone, args)
        self.branches = nn.ModuleList(self._get_branches(backbone, args))

    def _get_common_branch(self, backbone, args):
        return NotImplemented

    def _get_middle_subbranch_for(self, backbone, args, last_branch_class):

        return NotImplemented

    def _get_branches(self, backbone, args) -> list:

        branch_names = frozenset(args['branches'])
        branch_list = []

        for branch_name in branch_names:

            if 'global' == branch_name:

                middle_subbranch = self._get_middle_subbranch_for(backbone, args, GlobalBranch)
                global_branch = GlobalBranch(self, backbone, args, middle_subbranch.out_dim)

                branch_list.append(
                    Sequential(
                        middle_subbranch,
                        global_branch
                    )
                )

            if 'abd' == branch_name:
                middle_subbranch = self._get_middle_subbranch_for(backbone, args, ABDBranch)
                abd_branch = ABDBranch(self, backbone, args, middle_subbranch.out_dim)
                branch_list.append(
                    Sequential(
                        middle_subbranch,
                        abd_branch
                    )
                )

            if branch_name.startswith('np'):
                middle_subbranch = self._get_middle_subbranch_for(backbone, args, NPBranch)
                try:
                    part_num = int(branch_name[2:])
                except (TypeError, ValueError):
                    part_num = None
                np_branch = NPBranch(self, backbone, args, middle_subbranch.out_dim, part_num=part_num)
                branch_list.append(
                    Sequential(
                        middle_subbranch,
                        np_branch
                    )
                )

            if 'dan' == branch_name:
                middle_subbranch = self._get_middle_subbranch_for(backbone, args, DANBranch)
                dan_branch = DANBranch(self, backbone, args, middle_subbranch.out_dim)
                branch_list.append(
                    Sequential(
                        middle_subbranch,
                        dan_branch
                    )
                )

        assert len(branch_list) != 0, 'Should specify at least one branch.'
        return branch_list

    def backbone_modules(self):

        lst = [*self.common_branch.backbone_modules()]
        for branch in self.branches:
            lst.extend(branch.backbone_modules())

        return lst

    def forward(self, x, label=None):
        
        x, *intermediate_fmaps = self.common_branch(x, label)

        fmap_dict = defaultdict(list)
        fmap_dict['intermediate'].extend(intermediate_fmaps)

        predict_features, xent_features, triplet_features = [], [], []
        
        for branch in self.branches:

            #pdb.set_trace()
            arg_x = branch[0](x)
            predict, xent, triplet, fmap = branch[1](arg_x, label)
            predict_features.extend(predict)
            xent_features.extend(xent)
            triplet_features.extend(triplet)

            for name, fmap_list in fmap.items():
                fmap_dict[name].extend(fmap_list)

        fmap_dict = {k: tuple(v) for k, v in fmap_dict.items()}

        return torch.cat(predict_features, 1), tuple(xent_features),\
            tuple(triplet_features), fmap_dict


class Sequential(nn.Sequential):

    def backbone_modules(self):

        backbone_modules = []
        for m in self._modules.values():
            backbone_modules.append(m.backbone_modules())

        return backbone_modules


class GlobalBranch(nn.Module):

    def __init__(self, owner, backbone, args, input_dim):
        super().__init__()

        self.owner = weakref.ref(owner)

        self.input_dim = input_dim
        self.output_dim = args['global_dim']
        self.args = args
        self.num_classes = owner.num_classes
        
        self._init_fc_layer()
        if args['global_max_pooling']:
            self.avgpool = nn.AdaptiveMaxPool2d(1)
        else:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
        self._init_classifier()

        self.final = ArcMarginProduct(
            self.output_dim, self.num_classes, multi_task=False,\
                 s=30.0, m=0.5, device=device
        )
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def backbone_modules(self):

        return []

    def _init_classifier(self):

        #bn = nn.BatchNorm1d(self.output_dim)
        inference_stage = nn.Linear(self.output_dim, self.output_dim)
        classifier = nn.Linear(self.output_dim, self.num_classes)
        init_params(inference_stage)
        init_params(classifier)

        #self.bn = bn
        self.inference_stage = inference_stage
        self.classifier = classifier

    def _init_fc_layer(self):

        dropout_p = self.args['dropout']

        if dropout_p is not None:
            dropout_layer = [nn.Dropout(p=dropout_p)]
        else:
            dropout_layer = []

        fc = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
            nn.BatchNorm1d(self.output_dim),
            nn.ReLU(inplace=True),
            *dropout_layer
        )
        init_params(fc)

        self.fc = fc

    def forward(self, x, label=None):

        triplet, xent, predict = [], [], []

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        triplet.append(x)
        predict.append(x)
        '''
        x = self.inference_stage(x)
        x = self.classifier(x)
        '''
        x = self.final(x, label)
        x = self.logsoftmax(x)
        xent.append(x)

        return predict, xent, triplet, {}

class NPBranch(nn.Module):

    def __init__(self, owner, backbone, args, input_dim, part_num=None):
        super().__init__()

        self.owner = weakref.ref(owner)

        self.input_dim = input_dim
        self.output_dim = args['np_dim']
        self.args = args
        self.num_classes = owner.num_classes
        self.with_global = args['np_with_global']
        if part_num is None:
            part_num = args['np_np']
        self.part_num = subbranch_num = part_num
        if self.with_global:
            subbranch_num += 1

        self.fcs = nn.ModuleList([self._init_fc_layer() for i in range(subbranch_num)])
        if args['np_max_pooling']:
            self.avgpool = nn.AdaptiveMaxPool2d(1)
        else:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifiers = nn.ModuleList([self._init_classifier() for i in range(subbranch_num)])

    def backbone_modules(self):

        return []

    def _init_classifier(self):

        classifier = nn.Linear(self.output_dim, self.num_classes)
        init_params(classifier)

        return classifier

    def _init_fc_layer(self):

        dropout_p = self.args['dropout']

        if dropout_p is not None:
            dropout_layer = [nn.Dropout(p=dropout_p)]
        else:
            dropout_layer = []

        fc = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
            nn.BatchNorm1d(self.output_dim),
            nn.ReLU(inplace=True),
            *dropout_layer
        )
        init_params(fc)

        return fc

    def forward(self, x, label=None):

        triplet, xent, predict = [], [], []

        assert x.size(2) % self.part_num == 0,\
            "Height {} is not a multiplication of {}. Aborted.".format(x.size(2), self.part_num)
        margin = x.size(2) // self.part_num

        for p in range(self.part_num):
            x_sliced = self.avgpool(x[:, :, p * margin:(p + 1) * margin, :])
            x_sliced = x_sliced.view(x_sliced.size(0), -1)

            x_sliced = self.fcs[p](x_sliced)
            triplet.append(x_sliced)
            predict.append(x_sliced)
            x_sliced = self.classifiers[p](x_sliced)
            xent.append(x_sliced)

        if self.with_global:
            x_sliced = self.avgpool(x)
            x_sliced = x_sliced.view(x_sliced.size(0), -1)

            x_sliced = self.fcs[-1](x_sliced)
            triplet.append(x_sliced)
            predict.append(x_sliced)
            x_sliced = self.classifiers[-1](x_sliced)
            xent.append(x_sliced)

        return predict, xent, triplet, {}


class ABDBranch(nn.Module):

    def __init__(self, owner, backbone, args, input_dim):
        super().__init__()

        self.owner = weakref.ref(owner)

        self.input_dim = input_dim
        self.output_dim = args['abd_dim']
        self.args = args
        self.part_num = args['abd_np']
        self.num_classes = owner.num_classes

        self._init_reduction_layer()
        self._init_attention_modules()

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self._init_classifiers()


    def backbone_modules(self):

        return []

    def _init_classifiers(self):

        self.inference_stages = nn.ModuleList()
        self.finals = nn.ModuleList()
        self.logsoftmaxs = nn.ModuleList()

        '''
        for p in range(1, self.part_num + 1):
            #bn = nn.BatchNorm1d(self.output_dim)
            inference_stage = nn.Linear(self.output_dim, self.output_dim)
            init_params(inference_stage)
            #self.inference_stages.append(bn)
            self.inference_stages.append(inference_stage)
        '''

        for p in range(1, self.part_num+1):
            final = ArcMarginProduct(
                self.output_dim, self.num_classes, multi_task=False,\
                    s=30.0, m=0.5, device=device
            )
            self.finals.append(final)
            logsoftmax = nn.LogSoftmax(dim=1)
            self.logsoftmaxs.append(logsoftmax)

        self.classifiers = nn.ModuleList()

        for p in range(1, self.part_num + 1):
            classifier = nn.Linear(self.output_dim, self.num_classes)
            init_params(classifier)
            self.classifiers.append(classifier)

    def _init_reduction_layer(self):

        reduction = nn.Sequential(
            nn.Conv2d(self.input_dim, self.output_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.output_dim),
            nn.ReLU(inplace=True)
        )
        init_params(reduction)

        self.reduction = reduction

    def _init_attention_modules(self):

        args = self.args
        self.dan_module_names = set()
        DAN_module_names = {'cam', 'pam'} & set(args['abd_dan'])
        use_head = not args['abd_dan_no_head']
        self.use_dan = bool(DAN_module_names)

        before_module = get_attention_module_instance(
            'identity',
            self.output_dim,
            use_head=use_head
        )
        self.dan_module_names.add('before_module')
        self.before_module = before_module
        if use_head:
            init_params(before_module)

        if 'cam' in DAN_module_names:
            cam_module = get_attention_module_instance(
                'cam',
                self.output_dim,
                use_head=use_head
            )
            init_params(cam_module)
            self.dan_module_names.add('cam_module')
            self.cam_module = cam_module

        if 'pam' in DAN_module_names:
            pam_module = get_attention_module_instance(
                'pam',
                self.output_dim,
                use_head=use_head
            )
            init_params(pam_module)
            self.dan_module_names.add('pam_module')
            self.pam_module = pam_module

        sum_conv = nn.Sequential(
            nn.Dropout2d(0.1, False),
            nn.Conv2d(self.output_dim, self.output_dim, kernel_size=1)
        )
        init_params(sum_conv)
        self.sum_conv = sum_conv

    def forward(self, x, label=None):

        predict, xent, triplet = [], [], []
        fmap = defaultdict(list)

        x = self.reduction(x)

        assert x.size(2) % self.part_num == 0,\
            "Height {} is not a multiplication of {}. Aborted.".format(x.size(2), self.part_num)

        margin = x.size(2) // self.part_num
        for p in range(self.part_num):
            x_sliced = x[:, :, margin * p:margin * (p + 1), :]

            if self.use_dan:
                to_sum = []
                # module_name: str
                for module_name in self.dan_module_names:
                    x_out = getattr(self, module_name)(x_sliced)
                    to_sum.append(x_out)
                    fmap[module_name.partition('_')[0]].append(x_out)

                fmap_after = self.sum_conv(sum(to_sum))
                fmap['after'].append(fmap_after)

            else:

                fmap_after = x_sliced
                fmap['before'].append(fmap_after)
                fmap['after'].append(fmap_after)

            v = self.avgpool(fmap_after)
            v = v.view(v.size(0), -1)
            triplet.append(v)
            predict.append(v)
            '''
            v = self.inference_stages[p](v)
            v = self.classifiers[p](v)
            '''
            
            v = self.finals[p](v, label)
            v = self.logsoftmaxs[p](v)
            
            #v = self.classifiers[p](v)
            xent.append(v)

        return predict, xent, triplet, fmap


class DANBranch(nn.Module):

    def __init__(self, owner, backbone, args, input_dim):
        super().__init__()

        self.owner = weakref.ref(owner)

        self.input_dim = input_dim
        self.output_dim = args['dan_dim']
        self.args = args
        self.num_classes = owner.num_classes

        self._init_attention_modules()

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self._init_classifier()

    def backbone_modules(self):

        return []

    def _init_classifier(self):

        classifier = nn.Linear(len(self.dan_module_names) * self.output_dim, self.owner().num_classes)
        init_params(classifier)
        self.classifier = classifier

    def _init_attention_modules(self):

        args = self.args
        self.dan_module_names = set()
        DAN_module_names = {'cam', 'pam'} & set(args['dan_dan'])
        use_head = not args['dan_dan_no_head']
        self.use_dan = bool(DAN_module_names)

        before_module = get_attention_module_instance(
            'identity',
            self.output_dim,
            use_head=False
        )
        self.dan_module_names.add('before_module')
        self.before_module = before_module
        if use_head:
            init_params(before_module)

        if 'cam' in DAN_module_names:
            cam_module = get_attention_module_instance(
                'cam',
                self.input_dim,
                out_dim=self.output_dim,
                use_head=use_head
            )
            init_params(cam_module)
            self.dan_module_names.add('cam_module')
            self.cam_module = cam_module

        if 'pam' in DAN_module_names:
            pam_module = get_attention_module_instance(
                'pam',
                self.input_dim,
                out_dim=self.output_dim,

                use_head=use_head
            )
            init_params(pam_module)
            self.dan_module_names.add('pam_module')
            self.pam_module = pam_module

    def forward(self, x, label=None):

        predict, xent, triplet = [], [], []

        x = F.relu(x)

        feats = []
        # module_name: str
        for module_name in self.dan_module_names:
            x_out = getattr(self, module_name)(x)
            feats.append(x_out)

        v = self.avgpool(torch.cat(feats, 1))
        v = v.view(v.size(0), -1)
        triplet.append(v)
        predict.append(v)
        v = self.classifier(v)
        xent.append(v)

        return predict, xent, triplet, {}