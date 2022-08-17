import logging
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
from utils import test_single_volume
from networks.ScaleFormer import ScaleFormer
import argparse


def inference(args, model, test_save_path=None):
    db_test = Synapse_dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    with torch.no_grad():
        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            h, w = sampled_batch["image"].size()[2:]
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
            metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                          test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
            metric_list += np.array(metric_i)
            logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
        metric_list = metric_list / len(db_test)
        for i in range(1, args.num_classes):
            logging.info('Mean class %d mean_dice %f' % (i, metric_list[i-1]))
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
        logging.info("Testing Finished!")
        return performance


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', default=9)
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--root_path', type=str,
                        default='E:/project_TransUNet/data/Synapse/train_npz', help='root dir for data')
    parser.add_argument('--dataset', type=str,
                        default='Synapse', help='experiment_name')
    parser.add_argument('--list_dir', type=str,
                        default='./lists/lists_Synapse', help='list dir')
    parser.add_argument('--test_save_dir', default='./predictions_5', help='saving prediction as nii!')
    parser.add_argument("--volume_path", default="E:/project_TransUNet/data/Synapse/test_vol_h5/")
    parser.add_argument('--img_size', type=int,
                        default=224, help='input patch size of network input')
    parser.add_argument("--z_spacing", default=1)
    args = parser.parse_args()
    model = ScaleFormer(9)
    model.load_state_dict(torch.load(args.model_path))
    inference(args, model, args.test_save_dir)


