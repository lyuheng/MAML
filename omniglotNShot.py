from omniglot import Omniglot

import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np

class OmniglotNShot:

    def __init__(self, root, batchsz, n_way, k_shot, k_query, imgsz):

        self.resize = imgsz

        if not os.path.isfile(os.path.join(root, 'MAML/omniglot.npy')):
            self.x = Omniglot(root, transform=transforms.Compose([
                lambda x: Image.open(x).convert('L'),
                lambda x: x.resize((imgsz,imgsz)),
                lambda x: np.reshape(x, (imgsz,imgsz,1)),
                lambda x: np.transpose(x, [2,0,1]),
                lambda x: x/255.,
            ]))
            temp = dict()
            # len(temp) = 1623
            for (img, label) in self.x: # call __getitem__()
                if label in temp.keys():
                    temp[label].append(img)
                else:
                    temp[label] = [img]
            self.x = []
            for label, imgs in temp.items():
                self.x.append(np.array(imgs))
            self.x = np.array(self.x).astype(float)
            print('data shape: ', self.x.shape)
            temp = []  # free memory
            #np.save(os.path.join(root, 'MAML/omniglot.npy'), self.x) # [1623,20,84,84,1]
            print('write into onmiglot.npy successfully.')
        else:
            self.x = np.load(os.path.join(root,  'MAML/omniglot.npy'))
            print('load omniglot.npy successfully.')

        
        # TODO: keep training and test set distinct
        self.x_train, self.x_test = self.x[:1200], self.x[1200:]

        #self.normalization()

        self.batchsz = batchsz  # like our total dataset
        self.n_cls = self.x.shape[0]
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        assert (k_shot + k_query) <= 20

        self.indexes = {'train':0, 'test':0}
        self.datasets = {'train': self.x_train, 'test': self.x_test}
        print('DB train', self.x_train.shape, 'test', self.x_test.shape)

        self.datasets_cache = {'train': self.load_data_cache(self.datasets['train']),  # current epoch data cached
                               'test': self.load_data_cache(self.datasets['test'])}

    def normalization(self):
        """
        Normalizes our data, to have a mean of 0 and sdt of 1
        """
        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)
        # print("before norm:", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)
        self.x_train = (self.x_train - self.mean) / self.std
        self.x_test = (self.x_test - self.mean) / self.std

        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)


    def load_data_cache(self, data_pack):
        """
        Args:
        data_pack : [cls_num,20,28,28,1]
        return : [support_x, support_y, target_x, target_y]
        """
        setsz = self.k_shot * self.n_way
        querysz = self.k_query * self.n_way

        data_cache = []
        
        # in Mini_Imagenet, we do num_epoches//10000

        for sample in range(10):  # num of episodes

            x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
            for i in range(self.batchsz):  #  

                x_spt, y_spt, x_qry, y_qry = [], [], [], []
                selected_cls = np.random.choice(data_pack.shape[0], self.n_way, False)
                # select n_ways(classes) in this task
                for j, cur_class in enumerate(selected_cls):
                    # select (self.k_shot + self.k_query) in this class
                    selected_img = np.random.choice(20, self.k_shot + self.k_query, False)

                    # meta-training and meta-test
                    # add to support and query set
                    x_spt.append(data_pack[cur_class][selected_img[:self.k_shot]])
                    x_qry.append(data_pack[cur_class][selected_img[self.k_shot:]])
                    # simplify our label
                    y_spt.append([j for _ in range(self.k_shot)])
                    y_qry.append([j for _ in range(self.k_query)])

                # shuffle inside a batch
                perm = np.random.permutation(self.n_way * self.k_shot)
                x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot, 1, self.resize, self.resize)[perm]
                y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)[perm]
                perm = np.random.permutation(self.n_way * self.k_query)
                x_qry = np.array(x_qry).reshape(self.n_way * self.k_query, 1, self.resize, self.resize)[perm]
                y_qry = np.array(y_qry).reshape(self.n_way * self.k_query)[perm]

                # append [sptsz, 1, 84, 84] => [b, setsz, 1, 84, 84]
                x_spts.append(x_spt)
                y_spts.append(y_spt)
                x_qrys.append(x_qry)
                y_qrys.append(y_qry)


            # [b, setsz, 1, 28, 28]
            x_spts = np.array(x_spts).astype(np.float32).reshape(self.batchsz, setsz, 1, self.resize, self.resize)
            y_spts = np.array(y_spts).astype(np.int).reshape(self.batchsz, setsz)
            # [b, qrysz, 1, 28, 28]
            x_qrys = np.array(x_qrys).astype(np.float32).reshape(self.batchsz, querysz, 1, self.resize, self.resize)
            y_qrys = np.array(y_qrys).astype(np.int).reshape(self.batchsz, querysz)

            data_cache.append([x_spts, y_spts, x_qrys, y_qrys])

        return data_cache

    def next(self, mode='train'):
        """
        Gets next batch from the dataset with name.
        :param mode: The name of the splitting (one of "train", "val", "test")
        :return:
        """
        # update cache if indexes is larger cached num
        if self.indexes[mode] >= len(self.datasets_cache[mode]): # 10
            self.indexes[mode] = 0
            # generate new episode
            self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode])

        # first_time return: self.dataset_cache['train'][0]
        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1

        return next_batch
            

def main():
    import  time
    import  torch
    import  visdom
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    # plt.ion()
    #viz = visdom.Visdom(env='omniglot_view')

    db = OmniglotNShot(root='E:/meta_learning', batchsz=20, n_way=5, k_shot=5, k_query=15, imgsz=28)

    for i in range(10):
        x_spt, y_spt, x_qry, y_qry = db.next('train')
        # print(x_spt.shape)            
        # [b, setsz, h, w, c] => [b, setsz, c, w, h] => [b, setsz, 3c, w, h]
        """
        x_spt = torch.from_numpy(x_spt)
        x_qry = torch.from_numpy(x_qry)
        y_spt = torch.from_numpy(y_spt)
        y_qry = torch.from_numpy(y_qry)
        batchsz, setsz, c, h, w = x_spt.size()


        viz.images(x_spt[0], nrow=5, win='x_spt', opts=dict(title='x_spt'))
        viz.images(x_qry[0], nrow=15, win='x_qry', opts=dict(title='x_qry'))
        viz.text(str(y_spt[0]), win='y_spt', opts=dict(title='y_spt'))
        viz.text(str(y_qry[0]), win='y_qry', opts=dict(title='y_qry'))


        time.sleep(10)
        """
        
        x_spt = make_grid(torch.Tensor(x_spt[0]), nrow=10)
        x_qry = make_grid(torch.Tensor(x_qry[0]), nrow=10)

        plt.figure(1)
        plt.imshow(x_spt.transpose(2,0).numpy())
        plt.pause(0.5)

        plt.figure(2)
        plt.imshow(x_qry.transpose(2,0).numpy())
        plt.pause(0.5)

        time.sleep(5)
       
if __name__ == '__main__':
    pass