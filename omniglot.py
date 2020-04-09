import torch
import torch.utils.data as data
import os


class Omniglot(data.Dataset):

    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.all_items = find_classes(os.path.join(self.root, 'omniglot')) ####!!!
        self.idx_classes = index_classes(self.all_items)

    def __getitem__(self, index):
        filename = self.all_items[index][0]
        #########################################
        # add '/' between each element in list  #
        #########################################
        img = str.join('/', [self.all_items[index][2], filename]) # full address

        target = self.idx_classes[self.all_items[index][1]]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.all_items)


def find_classes(root_dir):
    retour = []
    for (root, dirs, files) in os.walk(root_dir):
        for f in files:
            if (f.endswith('png')):
                r = root.split('/')
                lr = len(r)
                ###########################
                # r[lr-2]: which language
                # r[lr-1]: which character
                ###########################
                retour.append(( f, r[lr-2]+'/'+r[lr-1], root ))
    print("== Found %d items" % len(retour))
    return retour

# affine index ot number {0,1,2,3, ...}
def index_classes(items):
    idx = {}
    for i in items:
        if i[1] not in idx:
            idx[i[1]] = len(idx)
    print('== Found %d classes' % len(idx))
    return idx


# for testing

# ROOT = 'E:/meta_learning'

# all_items = find_classes(os.path.join(ROOT, 'omniglot/'))
# index_classes(all_items)

# background:
# == Found 19280 items
# == Found 964 classes

# background + evaluation: 
# == Found 32460 items
# == Found 1623 classes
