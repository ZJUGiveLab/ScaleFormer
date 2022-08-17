import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
import matplotlib.pyplot as plt


def drawLoss(Loss, epoch_num, name, snapshot_path):
    fig1, ax1 = plt.subplots(figsize=(11, 8))
    ax1.plot(range(epoch_num + 1), Loss)
    ax1.set_title("Average trainset loss vs epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Current loss")
    plt.savefig(os.path.join(snapshot_path, 'loss_' + name + '_vs_epochs.png'))

    plt.clf()
    plt.close()


def trainer_synapse(args, model, snapshot_path):
    Loss_f = []

    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    
    for epoch_num in iterator:
        train_loss_f = 0

        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            o_f = model(image_batch)
            torch.cuda.empty_cache()
            loss_ce = ce_loss(o_f, label_batch[:].long())
            loss_dice = dice_loss(o_f, label_batch, softmax=True)
            train_loss_f += 0.5 * loss_ce.item() + 0.5 * loss_dice.item()

            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)


        logging.info('epoch %d :, lr %.6f, loss : %f, loss_ce: %f, max_dice=%f/%d' % (epoch_num, lr_, loss.item(), loss_ce.item(), max_dice, max_turn))

        Loss_f.append(train_loss_f / len(trainloader))
        drawLoss(Loss_f, epoch_num, name='f', snapshot_path=snapshot_path)
        if (epoch_num % args.save_interval == 0 and epoch_num != 0) or (epoch_num+args.loadNum >= max_epoch):
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num+args.loadNum) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            log_name = args.snapshot_path + "/eval/" + str(epoch_num+args.loadNum) + ".txt"

        if epoch_num >= max_epoch:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"
