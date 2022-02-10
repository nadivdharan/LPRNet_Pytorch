# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
test pretrained model.
Author: aiboy.wei@outlook.com .
'''

from data.load_data import CHARS, CHARS_DICT, LPRDataLoader
from PIL import Image, ImageDraw, ImageFont
from model.LPRNet import build_lprnet
# import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
import numpy as np
import argparse
import torch
import time
import cv2
import os
from tqdm import tqdm


MAX_TO_SHOW = 50


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--img_size', default=[120, 30], help='the image size')
    parser.add_argument('--test_img_dirs', default="./data/test", help='the test images path')
    parser.add_argument('--dropout_rate', default=0.5, help='dropout rate.')
    parser.add_argument('--lpr_max_len', default=10, type=int, help='license plate number max length.')
    parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size.')
    # parser.add_argument('--phase_train', default=False, type=bool, help='train or test phase flag.')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
    # parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
    parser.add_argument('--cuda', action='store_true', help='Use cuda to train model')
    parser.add_argument('--drop', action='store_true', help='Use dropoutt')
    parser.add_argument('--debug', action='store_true', help='Debug by printing predictions vs. labels')
    parser.add_argument('--crop', default=20, type=int, help='Number of pixels cropped from left part of image')
    # parser.add_argument('--show', default=False, type=bool, help='show test image and its predict result or not.')
    parser.add_argument('--show', action='store_true', help='show test image and its predict result or not.')
    parser.add_argument('--pretrained_model', default='./weights/Final_LPRNet_model.pth', help='pretrained base model')
    parser.add_argument('--save_dir', default='./preds', help='Location to save checkpoint models')

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
    labels = np.asarray(labels).flatten().astype(np.float32)

    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)

def test():
    args = get_parser()

    lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase='test', class_num=len(CHARS), dropout_rate=args.dropout_rate, drop=args.drop)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    lprnet.to(device)
    print("Successful to build network!")

    # load pretrained model
    if args.pretrained_model:
        lprnet.load_state_dict(torch.load(args.pretrained_model, map_location=torch.device(device)))
        print("load pretrained model successful!")
    else:
        print("[Error] Can't found pretrained mode, please check!")
        return False

    test_img_dirs = os.path.expanduser(args.test_img_dirs)
    test_dataset = LPRDataLoader(test_img_dirs.split(','), args.img_size, args.lpr_max_len, train=False, crop=args.crop)
    try:
        Greedy_Decode_Eval(lprnet, test_dataset, args)
    finally:
        cv2.destroyAllWindows()

def Greedy_Decode_Eval(Net, datasets, args):
    # TestNet = Net.eval()
    Net.eval()
    epoch_size = len(datasets) // args.test_batch_size
    batch_iterator = iter(DataLoader(datasets, args.test_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))

    Tp = 0
    Tn_1 = 0
    Tn_2 = 0
    showed = 0
    precision = 0
    lv_sim = 0 
    t1 = time.time()
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
        imgs = images.numpy().copy()

        if args.cuda:
            images = Variable(images.cuda())
        else:
            images = Variable(images)

        # forward
        prebs = Net(images)
        # greedy decode
        prebs = prebs.cpu().detach().numpy()
        preb_labels = list()
        for i in range(prebs.shape[0]):
            preb = prebs[i, :, :]
            preb_label = list()
            for j in range(preb.shape[1]):
                preb_label.append(np.argmax(preb[:, j], axis=0))
            if args.debug:
                print("Ground Befor Decode", ''.join([CHARS[int(x)] for x in preb_label]))
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
        Acc_2 = 0
        lv = 0
        for i, label in enumerate(preb_labels):
            # show image and its predict label
            if args.show and showed < MAX_TO_SHOW:
                show(imgs[i], label, targets[i], save_dir=args.save_dir)
                showed += 1
            from strsimpy.normalized_levenshtein import NormalizedLevenshtein
            lv_score = NormalizedLevenshtein().similarity(label, targets[i].tolist())
            lv += lv_score
            if args.debug:
                print("Ground Truth", ''.join([CHARS[int(x)] for x in targets[i]]))
                print("Predicted:  ", ''.join([CHARS[int(x)] for x in label]))
                print('-------------------------------')
            
            # Precision: mean true character recognition rate per sequence
            min_len = min(len(label), len(targets[i]))
            max_len = max(len(label), len(targets[i]))
            Acc_2 += np.sum([ x==y for (x, y) in zip(label[:min_len], targets[i][:min_len]) ]) / (max_len * len(preb_labels))
            
            # Accuracy: mean true character recognition rate pre sequence
            if len(label) != len(targets[i]):
                Tn_1 += 1
                continue
            if (np.asarray(targets[i]) == np.asarray(label)).all():
                Tp += 1
            else:
                Tn_2 += 1
        lv_sim += lv / len(preb_labels)
        precision += Acc_2
    precision /= epoch_size
    lv_sim /= epoch_size
    Acc = Tp * 1.0 / (Tp + Tn_1 + Tn_2)
    print("[Info] Test Accuracy: {} {} [{}:{}:{}:{}]".format(Acc, precision, Tp, Tn_1, Tn_2, (Tp+Tn_1+Tn_2)))
    print("[Info] Test Levenshtein Similarity: {}".format(lv_sim))
    t2 = time.time()
    print("[Info] Test Speed: {}s 1/{}]".format((t2 - t1) / len(datasets), len(datasets)))

def show(img, label, target, save_dir=None):
    import matplotlib.pyplot as plt
    if not save_dir:
        raise ValueError("Path to save predictions not provided")
    img = np.transpose(img, (1, 2, 0))  # C, H, W -->  H, W, C
    img *= 128.
    img += 127.5
    img = img.astype(np.uint8)
    img = img[..., [2, 1, 0]]  # BGR --> RGB

    lb = ""
    for i in label:
        lb += CHARS[i]
    tg = ""
    for j in target.tolist():
        tg += CHARS[int(j)]

    flag = "F"
    if lb == tg:
        flag = "T"
    
    # plt.figure(figsize=(img.shape[1], img.shape[0]))
    plt.imshow(img)
    # plt.title(F'Predicted plate: {lb}', size=28)
    plt.title(F'{lb}', size=28)
    plt.axis('off')
    save_name = os.path.join(save_dir, F'{tg}.jpg')
    print(F'Saving predictions to {save_name}')
    plt.savefig(save_name)
    return save_name

    # img = cv2.putText(img, lb, (0,16), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0, 0, 255), 1)
    # img = cv2ImgAddText(img, lb, (0, 0), textColor='black', textSize=8)
    # cv2.imwrite('/data/data/nadivd/ocr_predictions.jpg', img)
    # cv2.imshow("test", img)
    # print("target: ", tg, " ### {} ### ".format(flag), "predict: ", lb)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

def cv2ImgAddText(img, text, pos, textColor=(255, 0, 0), textSize=12):
    if (isinstance(img, np.ndarray)):  # detect opencv format or not
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("data/NotoSansCJK-Regular.ttc", textSize, encoding="utf-8")
    draw.text(pos, text, textColor, font=fontText)

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    test()
