import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.ScaleFormer import ScaleFormer as DualViT_seg
import shutil
from trainer_synapse import trainer_synapse
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='E:/project_TransUNet/data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=600000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=8, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--max_epochs', type=int,
                    default=600, help='maximum epoch number to train')
parser.add_argument('--snapshot_path', type=str,
                    default="../model/Synapse/CNNTrans/Ablation_Study/Network8127_backbone_low", help='vit_patches_size, default is 16')
parser.add_argument('--isDeep', type=bool,
                    default=False, help='vit_patches_size, default is 16')
parser.add_argument('--save_interval', type=int,
                    default=25, help='vit_patches_size, default is 16')
parser.add_argument('--start_save', type=int,
                    default=50, help='vit_patches_size, default is 16')
parser.add_argument('--isLoad', type=bool,
                    default=False, help='vit_patches_size, default is 16')

args = parser.parse_args()

if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': 'E:/project_TransUNet/data/Synapse/train_npz',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
        },
    }
    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
        
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = False

    if not os.path.exists(args.snapshot_path):
        os.makedirs(args.snapshot_path)
    if os.path.exists(args.snapshot_path + '/mycode'):
        shutil.rmtree(args.snapshot_path + '/mycode')
    shutil.copytree('.', args.snapshot_path + '/mycode', shutil.ignore_patterns(['.git', '__pycache__']))

    if not os.path.exists(args.snapshot_path + "/eval"):
        os.makedirs(args.snapshot_path + "/eval")

    net = DualViT_seg(n_classes=args.num_classes).cuda()
    print('# generator parameters:', 1.0 * sum(param.numel() for param in net.parameters()) / 1000000)

    ##########################
    if args.isLoad:
        saved_state_dict = torch.load(args.loadPath)
        new_params = net.state_dict().copy()
        for name, param in new_params.items():
            if name in saved_state_dict and param.size() == saved_state_dict[name].size():
                new_params[name].copy_(saved_state_dict[name])
        net.load_state_dict(new_params)
    ##########################

    trainer = {'Synapse': trainer_synapse}
    trainer[dataset_name](args, net, args.snapshot_path)
