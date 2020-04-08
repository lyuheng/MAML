import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F


import numpy as np

from imagenet_learner import Learner
from copy import deepcopy

class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args):
        super().__init__()

        # two learning rates for \theta and \phi
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr  

        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.net = Learner(self.n_way)  # base meta net
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)

    def clip_grad_by_norm(self, grad, max_norm):
        pass

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """
        Args:
        x_sqt : [b(task_batch_number for training), setsz, c_, h, w]
        y_sqt : [b, setsz]
        x_qry : [b(task_batch_number for testing), querysz, c_, h, w]
        y_qry : [b, querysz]
        """
        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step+1)]
        corrects = [0 for _ in range(self.update_step+1)]

        net_update = deepcopy(self.net)

        for i in range(task_num):

            logits = self.net(x_spt[i])
            loss = F.cross_entropy(logits, y_spt[i])
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weight = list(map(lambda p: p[1]-self.update_lr*p[0], zip(grad, self.net.parameters())))
            net_update.assign_weight(fast_weight)

            with torch.no_grad():
                logits_q = self.net(x_qry[i])
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] += correct

            with torch.no_grad():
                logits_q = net_update(x_qry[i])
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[1] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] += correct

            for k in range(1, self.update_step):
                # update net_update
                logits = net_update(x_spt[i])
                loss = F.cross_entropy(logits, y_spt[i])
                grad = torch.autograd.grad(loss, net_update.parameters())
                fast_weight = list(map(lambda p: p[1]-self.update_lr*p[0], zip(grad, net_update.parameters())))
                net_update.assign_weight(fast_weight)
                
                # test on qry set
                logits_q = net_update(x_qry[i])
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[k+1] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[k+1] += correct
        
        # end of all tasks in this batch
        # sum over all losses on >>> query set <<< across all tasks
        loss_q = losses_q[-1]/task_num

        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()

        accs = np.array(corrects)/(querysz*task_num)

        del net_update

        return accs

    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """
        use the initalized parameters we get to train on a new task

        Args:
        x_sqt : [setsz, c_, h, w]
        y_sqt : [setsz]
        x_qry : [querysz, c_, h, w]
        y_sqt : [querysz]
        """

        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test+1)]

        net = deepcopy(self.net)
        net_update = deepcopy(self.net)

        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weight = list(map(lambda p: p[1]-self.update_lr*p[0], zip(grad, net.parameters())))
        net_update.assign_weight(fast_weight)

        with torch.no_grad():
            logits_q = net(x_qry)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] += correct
        with torch.no_grad():
            logits_q = net_update(x_qry)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] += correct

        for k in range(1, self.update_step_test): ###!!!
            # update net_update
            logits = net_update(x_spt)
            loss = F.cross_entropy(logits, y_spt)
            grad = torch.autograd.grad(loss, net_update.parameters())
            fast_weight = list(map(lambda p: p[1]-self.update_lr*p[0], zip(grad, net_update.parameters())))
            net_update.assign_weight(fast_weight)
            
            # test on qry set
            logits_q = net_update(x_qry)
            loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()
                corrects[k+1] += correct
        
        del net
        del net_update
        accs = np.array(corrects)/querysz
        return accs

def main():
    pass

if __name__ == '__main__':
    main()
        




