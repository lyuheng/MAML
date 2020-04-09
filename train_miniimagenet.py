"""
MAML is hard to train!
"""

import os, torch

import numpy as np
from miniImagenet import Mini_Imagenet
from imagenet_meta import Meta
import torch.utils.data as data
import torch.optim as optim
import random, sys, pickle
import argparse


def main():
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    device = torch.device('cuda')
    maml = Meta(args).to(device)
    
    mini = Mini_Imagenet(root='E:/meta_learning/', mode='train', n_way=args.n_way, k_shot=args.k_spt, \
        k_query=args.k_qry, batchsz=10000, resize=args.imgsz)
    mini_test = Mini_Imagenet(root='E:/meta_learning/', mode='test', n_way=args.n_way, k_shot=args.k_spt, \
        k_query=args.k_qry, batchsz=100, resize=args.imgsz)
    
    for epoch in range(args.epoch//10000):
        db = data.DataLoader(mini, args.task_num, shuffle=True)

        # 10000 tasks in db
        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), \
                x_qry.to(device), y_qry.to(device)

            accs = maml(x_spt, y_spt, x_qry, y_qry) # accuracy on train set(average over a batch of tasks)
            if step % 30 == 0:
                print('step:', step, '\t training acc', accs)
            if step % 500 == 0:
                db_test = data.DataLoader(mini_test, 1, shuffle=True)
                accs_all_test = []

                for x_spt, y_spt, x_qry, y_qry in db_test:
                    # remove task_batch=1
                    x_spt = x_spt.squeeze(0).to(device) 
                    y_spt = y_spt.squeeze(0).to(device)
                    x_qry = x_qry.squeeze(0).to(device)
                    y_qry = y_qry.squeeze(0).to(device)
                    
                    accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
                    accs_all_test.append(accs)

                # average accuracy over test tasks on every round
                # [b, update_step+1]
                accs = np.array(accs_all_test).mean(axis=0).astype(np.float64)
                print('Test acc', accs)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    
    argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('-imgsz', type=int, help='img size', default=84)
    argparser.add_argument('--task_num', type=int, help='meta batch size', default=2) # lower task_num for out of GPU memory
    argparser.add_argument('--meta_lr', type=float, help='meta-level lr', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level lr', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='finetunning update steps', default=10)

    args = argparser.parse_args()

    main()


