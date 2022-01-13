# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
Pytorch implementation for LPRNet.
Author: aiboy.wei@outlook.com .
'''

from data.load_data import CHARS, CHARS_DICT, LPRDataLoader
from model.LPRNet import build_lprnet
# from model.LPRNet_orig import build_lprnet
# import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
from torchsummary import summary
import numpy as np
import argparse
import torch
import time
import os
import yaml
import pandas as pd
from tqdm import tqdm


log_dict = {'iteration': list(),
            'epoch': list(),
            'training loss': list(),
            'train_loss': list(),
            'test_loss': list(),
            'train accuracy': list(),
            'test accuracy': list(),
            'mean_acc_train': list(),
            'mean_acc_test': list()}

def sparse_tuple_for_ctc(T_length, lengths):
    input_lengths = []
    target_lengths = []

    for ch in lengths:
        input_lengths.append(T_length)
        target_lengths.append(ch)

    return tuple(input_lengths), tuple(target_lengths)

def adjust_learning_rate(optimizer, cur_epoch, base_lr, lr_schedule):
    """
    Sets the learning rate
    """
    lr = 0
    for i, e in enumerate(lr_schedule):
        if cur_epoch < e:
            lr = base_lr * (0.1 ** i)
            # lr = base_lr * (0.5 ** i)

            # print("***** ADJUSTING ****", lr)
            break
    if lr == 0:
        lr = base_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--max_epoch', type=int, default=15, help='epoch to train the network')
    parser.add_argument('--img_size', default=[96, 24], help='the image size')
    parser.add_argument('--train_img_dirs', default="~/workspace/trainMixLPR", help='the train images path')
    parser.add_argument('--test_img_dirs', default="~/workspace/testMixLPR", help='the test images path')
    parser.add_argument('--dropout_rate', default=0.5, help='dropout rate.')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='base value of learning rate.')
    parser.add_argument('--lpr_max_len', default=10, type=int, help='license plate number max length.')
    parser.add_argument('--train_batch_size', type=int, default=128, help='training batch size.')
    parser.add_argument('--test_batch_size', type=int, default=120, help='testing batch size.')
    # parser.add_argument('--phase_train', default=True, type=bool, help='train or test phase flag.')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=False, type=bool, help='Use cuda to train model')
    parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
    parser.add_argument('--save_interval', default=2000, type=int, help='epoch interval for save model state dict')
    parser.add_argument('--test_interval', default=2000, type=int, help='epoch interval for evaluate')
    parser.add_argument('--print_interval', default=2000, type=int, help='epoch interval for info print')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=2e-5, type=float, help='Weight decay for SGD')
    # parser.add_argument('--lr_schedule', default=[4, 8, 12, 14, 16], help='schedule for learning rate.')
    # parser.add_argument('--lr_schedule', type=int, nargs='+', default=[1e9], help='schedule for learning rate.')
    parser.add_argument('--lr_schedule', type=int, nargs='+', default=[10], help='schedule for learning rate.')
    parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')
    # parser.add_argument('--pretrained_model', default='./weights/Final_LPRNet_model.pth', help='pretrained base model')
    parser.add_argument('--pretrained_model', default='', help='pretrained base model')

    args = parser.parse_args()

    return args

def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
    labels = np.asarray(labels).flatten().astype(np.int)

    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)


def update_d(dd, *args):
    for i, item in enumerate(args):
        key = list(dd.keys())[i]
        dd[key].append(item)


def log_training(path, *args):
    if len(log_dict) != len(args):
        raise ValueError(F"{len(args)} values were given to log but expecting only {len(log_dict)}")
    update_d(log_dict, *args)
    df = pd.DataFrame(log_dict).set_index(list(log_dict.keys())[0])
    df.to_csv(os.path.join(path, 'training.log'))


def get_test_loss(net, dataset, batch_size, num_workers, T_length):
    # compute the mean loss over the test set
    net.eval()
    epoch_size = len(dataset) // batch_size
    batch_iterator = iter(DataLoader(dataset, batch_size,
                                        shuffle=False, num_workers=num_workers,
                                        collate_fn=collate_fn))
    ctc_loss = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean') # reduction: 'none' | 'mean' | 'sum'
    loss_val = 0
    for _ in range(epoch_size):
        images, labels, lengths = next(batch_iterator)
        input_lengths, target_lengths = sparse_tuple_for_ctc(T_length, lengths)
        logits = net(images)
        log_probs = logits.permute(2, 0, 1) # for ctc loss: T x N x C
        # print(labels.shape)
        log_probs = log_probs.log_softmax(2)
        loss = ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)
        loss_val += loss.item()

    net.train()
    return loss_val / epoch_size

def train():
    args = get_parser()

    T_length = 21 # args.lpr_max_len
    epoch = 0 + args.resume_epoch
    loss_val = 0
    best_acc = 0
    best_acc_train = 0
    train_loss = 0
    test_loss = 0
    mean_acc_test = 0
    mean_acc_train = 0

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    device = torch.device("cuda:0" if args.cuda else "cpu")
    lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase='train', class_num=len(CHARS), dropout_rate=args.dropout_rate, device=device)
    lprnet.to(device)
    print("Successful to build network!")

    # load pretrained model
    if args.pretrained_model:
        lprnet.load_state_dict(torch.load(args.pretrained_model))
        print("load pretrained model successful!")
    else:
        def xavier(param):
            nn.init.xavier_uniform(param)

        def weights_init(m):
            for key in m.state_dict():
                if key.split('.')[-1] == 'weight':
                    if 'conv' in key:
                        nn.init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                    if 'bn' in key:
                        m.state_dict()[key][...] = xavier(1)
                elif key.split('.')[-1] == 'bias':
                    m.state_dict()[key][...] = 0.01

        # lprnet.backbone.apply(weights_init)
        lprnet.stage1.apply(weights_init)
        lprnet.stage2.apply(weights_init)
        lprnet.stage3.apply(weights_init)
        lprnet.stage4.apply(weights_init)
        lprnet.downsample1.apply(weights_init)
        lprnet.downsample2.apply(weights_init)
        lprnet.container.apply(weights_init)
        print("initial net weights successful!")

    # define optimizer
    # optimizer = optim.SGD(lprnet.parameters(), lr=args.learning_rate,
                        #   momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.RMSprop(lprnet.parameters(), lr=args.learning_rate, alpha = 0.9, eps=1e-08,
                         momentum=args.momentum, weight_decay=args.weight_decay)
    train_img_dirs = os.path.expanduser(args.train_img_dirs)
    test_img_dirs = os.path.expanduser(args.test_img_dirs)
    train_dataset = LPRDataLoader(train_img_dirs.split(','), args.img_size, args.lpr_max_len)
    test_dataset = LPRDataLoader(test_img_dirs.split(','), args.img_size, args.lpr_max_len)

    epoch_size = len(train_dataset) // args.train_batch_size # batches per epoch
    max_iter = args.max_epoch * epoch_size # max batches

    ctc_loss = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean') # reduction: 'none' | 'mean' | 'sum'

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    # save final parameters
    # torch.save(lprnet.state_dict(), args.save_folder + 'Final_LPRNet_model.pth')
    print(summary(lprnet, (3,args.img_size[0],args.img_size[1])))
    # import ipdb; ipdb.set_trace()

    with open(os.path.join(args.save_folder, 'opt.yaml'), 'w') as f:
        yaml.dump(args, f, sort_keys=False)

    for iteration in range(start_iter, max_iter): # iteration = batch number
        epoch_iter = iteration % epoch_size
        if epoch_iter == 0:
            # if epoch > 0:
            #     train_loss = loss_val / epoch_size  # mean loss per batch
            #     with torch.no_grad():
            #         test_loss = get_test_loss(lprnet, train_dataset,
            #                                   args.train_batch_size, args.num_workers,
            #                                   T_length)
                

            # create batch iterator
            batch_iterator = iter(DataLoader(train_dataset, args.train_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))
            loss_val = 0
            epoch += 1

        start_time = time.time()
        # load train data
        images, labels, lengths = next(batch_iterator)
        # labels = np.array([el.numpy() for el in labels]).T
        # print(labels)
        # get ctc parameters
        input_lengths, target_lengths = sparse_tuple_for_ctc(T_length, lengths)
        # update lr
        lr = adjust_learning_rate(optimizer, epoch, args.learning_rate, args.lr_schedule)

        if args.cuda:
            images = Variable(images, requires_grad=False).cuda()
            labels = Variable(labels, requires_grad=False).cuda()
        else:
            images = Variable(images, requires_grad=False)
            labels = Variable(labels, requires_grad=False)

        # forward
        logits = lprnet(images)
        log_probs = logits.permute(2, 0, 1) # for ctc loss: T x N x C
        # print(labels.shape)
        log_probs = log_probs.log_softmax(2).requires_grad_()
        # log_probs = log_probs.detach().requires_grad_()
        # print(log_probs.shape)
        # backprop
        optimizer.zero_grad()
        loss = ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)
        if loss.item() == np.inf:
            continue
        loss.backward()
        optimizer.step()
        loss_val += loss.item()
        end_time = time.time()
       
        if iteration !=0 and iteration % args.save_interval == 0:
            torch.save(lprnet.state_dict(), os.path.join(args.save_folder , 'LPRNet_' + '_iteration_' + repr(iteration) + '.pth'))

        if (iteration + 1) % args.test_interval == 0:
            with torch.no_grad():
                print('*** Evaluating on train set... ***')
                acc_train, mean_acc_train, train_loss = Greedy_Decode_Eval(lprnet, train_dataset,
                                                           args.train_batch_size, args,
                                                           T_length)#, debug='train')
                print('*** Evaluating on test set... ***')
                acc, mean_acc_test, test_loss = Greedy_Decode_Eval(lprnet, test_dataset,
                                                    args.test_batch_size, args,
                                                    T_length)#, debug='test')
            # lprnet.train() # should be switch to train mode
            best_acc_train = acc_train if acc_train > best_acc_train else best_acc_train
            # log_training(args.save_folder, epoch, iteration, loss.item(), acc_train, acc)
            log_training(args.save_folder, iteration, epoch,
                         loss_val/(epoch_iter + 1),
                         train_loss, test_loss,
                         acc_train, acc,
                         mean_acc_train, mean_acc_test)
            if acc > best_acc:
                print(F'New Best! {acc}')
                torch.save(lprnet.state_dict(), args.save_folder + 'Best_LPRNet_model.pth')
                best_acc = acc

        if iteration % args.print_interval == 0:
        # if epoch %  args.print_epoch == 0:
            print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(epoch_iter) + '/' + repr(epoch_size)
                  + '|| Totel iter ' + repr(iteration) + '|| Loss: %.4f|| ' % (loss.item())
                  + '|| Training Loss: %.4f|| ' % (loss_val / (epoch_iter + 1))
                  + '|| Train Loss: %.4f|| ' % train_loss
                  + '|| Tests Loss: %.4f|| ' % test_loss
                  + '|| Mean Acc.: %.2f || ' % mean_acc_test
                  + '|| Mean Train Acc.: %.2f || ' % mean_acc_train
                  + '|| Best Acc.: %.2f || ' % best_acc
                  + '|| Best Train Acc.: %.2f || ' % best_acc_train
                  + '|| Batch time: %.4f sec. ||' % (end_time - start_time) + 'LR: %.8f' % (lr))
    # final test
    print("Final test Accuracy:")
    Greedy_Decode_Eval(lprnet, test_dataset, args.test_batch_size, args, T_length)

    # save final parameters
    torch.save(lprnet.state_dict(), args.save_folder + 'Final_LPRNet_model.pth')

def Greedy_Decode_Eval(Net, datasets, batch_size, args, T_length, debug=None):#, mode='test'):
    # TestNet = Net.eval()
    # assert mode in ['test', 'train'], F"Unrecognized mode {mode}. Please choose from ['test', 'train']"
    Net.eval()
    
    epoch_size = len(datasets) // batch_size
    batch_iterator = iter(DataLoader(datasets, batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))
    # batch_size = args.test_batch_size if 'test' in mode else args.train_batch_size
    # epoch_size = len(datasets) // batch_size
    # batch_iterator = iter(DataLoader(datasets, batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))

    Tp = 0
    Tn_1 = 0
    Tn_2 = 0
    loss_val = 0
    mean_Acc_2 = 0
    ctc_loss = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean') # reduction: 'none' | 'mean' | 'sum'
    t1 = time.time()
    # for i in range(epoch_size):
    for _ in tqdm(range(epoch_size)):
        # load train data
        images, labels, lengths = next(batch_iterator)
        start = 0
        targets = []
        for length in lengths:
            label = labels[start:start+length]
            targets.append(label)
            start += length
        targets = np.array([el.numpy() for el in targets])

        if args.cuda:
            images = Variable(images.cuda())
        else:
            images = Variable(images)

        # forward
        prebs = Net(images)

        # get loss
        # with torch.no_grad():
        log_probs = prebs.permute(2, 0, 1) # for ctc loss: T x N x C
        log_probs = log_probs.log_softmax(2)
        input_lengths, target_lengths = sparse_tuple_for_ctc(T_length, lengths)
        loss = ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)
        # if loss.item() == np.inf:
            # continue
        loss_val += loss.item() / epoch_size
        if debug:
            print(F"*** {debug} in eval : {loss.item()}, {loss_val}")

        # greedy decode
        prebs = prebs.cpu().detach().numpy()
        preb_labels = list()
        for i in range(prebs.shape[0]):
            preb = prebs[i, :, :]
            preb_label = list()
            for j in range(preb.shape[1]):
                preb_label.append(np.argmax(preb[:, j], axis=0))
            no_repeat_blank_label = list()
            pre_c = preb_label[0]
            if pre_c != len(CHARS) - 1:
                no_repeat_blank_label.append(pre_c)
            for c in preb_label: # dropout repeate label and blank label
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            preb_labels.append(no_repeat_blank_label)
        
        # accuracy
        Acc_2 = 0
        for i, label in enumerate(preb_labels):
            min_len = min(len(label), len(targets[i]))
            max_len = max(len(label), len(targets[i]))
            
            Acc_2 += np.sum([ x==y for (x, y) in zip(label[:min_len], targets[i][:min_len]) ]) / (max_len * len(preb_labels))

            if len(label) != len(targets[i]):
                Tn_1 += 1
                continue
            if (np.asarray(targets[i]) == np.asarray(label)).all():
                Tp += 1
            else:
                Tn_2 += 1
        mean_Acc_2 += Acc_2
    mean_Acc_2 /= epoch_size
    try:
        Acc = Tp * 1.0 / (Tp + Tn_1 + Tn_2)
    except ZeroDivisionError:
        Acc = 0
    print("[Info] Accuracy: {} {} [{}:{}:{}:{}]".format(Acc, mean_Acc_2, Tp, Tn_1, Tn_2, (Tp+Tn_1+Tn_2)))
    t2 = time.time()
    print("[Info] Speed: {}s 1/{}]".format((t2 - t1) / len(datasets), len(datasets)))
    
    Net.train()

    return Acc, mean_Acc_2, loss_val

if __name__ == "__main__":
    train()
