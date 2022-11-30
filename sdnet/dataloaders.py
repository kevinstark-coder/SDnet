from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset,DataLoader
import
from mypath import Path
from glob import glob
import random
import copy


def trainval_loader(data_path,batch_size):
    trainset = SingleDataset(data_path,"train")
    valset = SingleDataset(data_path,"val")
    train_loaders = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loaders = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    return train_loaders,val_loaders

class SingleDataset(Dataset):
    def __init__(self, data_path, phase):
        self.root = os.path.join(data_path, phase)
        self.test_Atarget = os.path.join(data_path, phase,'target')
        self.paths_A = os.listdir(self.root)
        self.boundingbox = []
        self.index_A = [0]
        for dir in self.paths_A:
            Atarget_path = os.path.join(self.test_Atarget, dir)
            Atarget_img = load_nifty_volume_as_array2(Atarget_path)
            boundingbox = get_ND_bounding_box(Atarget_img, 60)
            self.boundingbox.append(boundingbox)
            idex = boundingbox[1][0] - boundingbox[0][0] + self.index_A[-1]
            # idex = boundingbox[1]-boundingbox[0]+self.index_A[-1]
            self.index_A.append(idex)
        self.crop_size = 256
        self.load_size = 288
        self.test = self.index_A[-1]

    def __getitem__(self, index):
        index_A = index % self.test
        id_A = int(np.where((np.array(self.index_A)) <= index_A)[0].max())
        if (id_A == len(self.paths_A)): id_A = id_A - 1
        A_path = os.path.join(self.root, self.paths_A[id_A])
        slice_A = index_A - self.index_A[id_A]
        volume_A = load_nifty_volume_as_array2(A_path)
        volume_A = crop_ND_volume_with_bounding_box(volume_A, self.boundingbox[id_A][0], self.boundingbox[id_A][1])
        A = volume_A[slice_A]
        Atarget_path = os.path.join(self.test_Atarget, self.paths_A[id_A])
        volume_Atarget = load_nifty_volume_as_array2(Atarget_path)
        volume_Atarget = crop_ND_volume_with_bounding_box(volume_Atarget, self.boundingbox[id_A][0],
                                                          self.boundingbox[id_A][1])
        A_gt = volume_Atarget[slice_A]
        img, gt = get_test_transform2(A, A_gt, self.load_size, self.crop_size)
        # img = torch.from_numpy(img.copy()).unsqueeze(0).float()
        # gt = torch.from_numpy(gt.copy()).unsqueeze(0).float()
        return img, gt

    def __len__(self):
        return self.test


def get_test_transform2(A, A_gt, load_size, crop_size, grayscale=False, method=Image.BICUBIC, convert=True):
    w, h = A.shape
    A[A > 240] = 240
    A[A < -160] = -160
    A = resize_3D_volume_to_given_shape(A, (load_size, load_size), 3)
    A_gt = resize_3D_volume_to_given_shape(A_gt, (load_size, load_size), 0)
    A = (A - A.min()) / (A.max() - A.min())
    if (A_gt.max() - A_gt.min() > 0):
        A_gt = (A_gt - A_gt.min()) / (A_gt.max() - A_gt.min())

    A = A.astype(np.float32)

    A_gt = A_gt.astype(np.float32)

    A, A_gt = center_crop(crop_size, A, A_gt)

    image_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    A = image_transform(A)
    mask_transform = transforms.Compose([transforms.ToTensor()])
    A_gt = mask_transform(A_gt)
    return A, A_gt

def center_crop(crop_size, image, label):
    w, h = image.shape
    if h < crop_size or w < crop_size:
        padh = crop_size - h if h < crop_size else 0
        padw = crop_size - w if w < crop_size else 0
        image = np.pad(image,((padw//2, padw - padw//2), (padh//2, padh- padh//2)), 'constant', constant_values=0)
        label = np.pad(label,((padw//2, padw - padw//2), (padh//2, padh- padh//2)), 'constant', constant_values=0)
        x1 = int(round((w+padw - crop_size) / 2.))
        y1 = int(round((h+padh - crop_size) / 2.))
        image = image[x1:x1+crop_size, y1:y1+crop_size]
        label = label[x1:x1+crop_size, y1:y1+crop_size]
    else:
        x1 = int(round((w - crop_size) / 2.))
        y1 = int(round((h - crop_size) / 2.))
        image = image[x1:x1+crop_size, y1:y1+crop_size]
        label = label[x1:x1+crop_size, y1:y1+crop_size]

def resize_3D_volume_to_given_shape(volume, out_shape, order = 3):
    """
    resize an nd volume to a given shape
    inputs:
        volume: the input nd volume, an nd array
        out_shape: the desired output shape, a list
        order: the order of interpolation
    outputs:
        out_volume: the reized nd volume with given shape
    """
    shape0=volume.shape
    assert(len(shape0) == len(out_shape))
    scale = [(out_shape[i] + 0.0)/shape0[i] for i in range(len(shape0))]
    out_volume = ndimage.interpolation.zoom(volume, scale, order = order)
    return out_volume

def load_nifty_volume_as_array2(filename, with_header=False, is_label=False):
    """
    load nifty image into numpy array, and transpose it based on the [z,y,x] axis order
    The output array shape is like [Depth, Height, Width]
    inputs:
        filename: the input file name, should be *.nii or *.nii.gz
        with_header: return affine and hearder infomation
    outputs:
        data: a numpy data array
    """
    img = nibabel.load(filename)
    data = img.get_data()
    data = np.transpose(data, [2, 1, 0])
    if (with_header):
        return data, img.affine, img.header
    else:
        return data

def crop_ND_volume_with_bounding_box(volume, min_idx, max_idx):
    """
    crop/extract a subregion form an nd image.
    """
    dim = len(volume.shape)
    assert (dim >= 2 and dim <= 5)
    '''if(len([min_idx]) == 1):
        output = volume[np.ix_(range(min_idx, max_idx + 1))]'''
    if (dim == 2):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1))]
    elif (dim == 3):
        output = volume[np.ix_(range(min_idx[0], max_idx[0]),
                               range(min_idx[1], max_idx[1]),
                               range(min_idx[2], max_idx[2]))]
    elif (dim == 4):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1),
                               range(min_idx[3], max_idx[3] + 1))]
    elif (dim == 5):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1),
                               range(min_idx[3], max_idx[3] + 1),
                               range(min_idx[4], max_idx[4] + 1))]
    else:
        raise ValueError("the dimension number shoud be 2 to 5")
    return output

def get_ND_bounding_box(label, margin):
    """
    get the bounding box of the non-zero region of an ND volume
    """
    input_shape = label.shape
    if (type(margin) is int):
        margin = [margin] * len(input_shape)
    assert (len(input_shape) == len(margin))
    indxes = np.nonzero(label)
    # print(indxes)
    idx_min = []
    idx_max = []
    for i in range(len(input_shape)):  # de dao gan zang shape
        idx_min.append(indxes[i].min())
        idx_max.append(indxes[i].max())

    for i in range(1,
                   len(input_shape)):  # xiang wai kuo zhang margin pixels, ru guo chao guo boudary, jiuyong boundary
        idx_min[i] = max(idx_min[i] - margin[i], 0)
        idx_max[i] = min(idx_max[i] + margin[i], input_shape[i] - 1)
    r = 0  # shang xia cai jian 1/3
    a = idx_min[0]
    b = idx_max[0]
    if (margin == 30):
        idx_min[0] = int((b - a) * r + a)
        idx_max[0] = int(b - (b - a) * r)
    else:
        idx_min[0] = int((b - a) * r + a)
    return idx_min, idx_max
# class FundusSegmentation(Dataset):
#     """
#     Fundus segmentation dataset
#     including 5 domain dataset
#     one for test others for training
#     """
#
#     def __init__(self,
#                  base_dir=Path.db_root_dir('fundus'),
#                  phase='train',
#                  splitid=[2, 3, 4],
#                  transform=None,
#                  state='train',
#                  ):
#         """
#         :param base_dir: path to VOC dataset directory
#         :param split: train/val
#         :param transform: transform to apply
#         """
#         # super().__init__()
#         self.state = state
#         self._base_dir = base_dir
#         self.image_list = []
#         self.phase = phase
#         self.image_pool = {'DGS':[], 'REF':[], 'RIM':[], 'REF_val':[]}
#         self.label_pool = {'DGS':[], 'REF':[], 'RIM':[], 'REF_val':[]}
#         self.img_name_pool = {'DGS':[], 'REF':[], 'RIM':[], 'REF_val':[]}
#
#         self.flags_DGS = ['gd', 'nd']
#         self.flags_REF = ['g', 'n']
#         self.flags_RIM = ['G', 'N', 'S']
#         self.flags_REF_val = ['V']
#         self.splitid = splitid
#         SEED = 1212
#         random.seed(SEED)
#         for id in splitid:
#             self._image_dir = os.path.join(self._base_dir, 'Domain'+str(id), phase, 'ROIs/image/')
#             print('==> Loading {} data from: {}'.format(phase, self._image_dir))
#
#             imagelist = glob(self._image_dir + '*.png')
#             for image_path in imagelist:
#                 gt_path = image_path.replace('image', 'mask')
#                 self.image_list.append({'image': image_path, 'label': gt_path})
#
#         self.transform = transform
#         self._read_img_into_memory()
#         for key in self.image_pool:
#             if len(self.image_pool[key]) < 1:
#                 del self.image_pool[key]
#                 del self.label_pool[key]
#                 del self.img_name_pool[key]
#                 break
#         for key in self.image_pool:
#             if len(self.image_pool[key]) < 1:
#                 del self.image_pool[key]
#                 del self.label_pool[key]
#                 del self.img_name_pool[key]
#                 break
#         for key in self.image_pool:
#             if len(self.image_pool[key]) < 1:
#                 del self.image_pool[key]
#                 del self.label_pool[key]
#                 del self.img_name_pool[key]
#                 break
#         # Display stats
#         print('-----Total number of images in {}: {:d}'.format(phase, len(self.image_list)))
#
#     def __len__(self):
#         max = -1
#         for key in self.image_pool:
#              if len(self.image_pool[key])>max:
#                  max = len(self.image_pool[key])
#         return max
#
#     def __getitem__(self, index):
#         if self.phase != 'test':
#             sample = []
#             for key in self.image_pool:
#                 domain_code = list(self.image_pool.keys()).index(key)
#                 index = np.random.choice(len(self.image_pool[key]), 1)[0]
#                 _img = self.image_pool[key][index]
#                 _target = self.label_pool[key][index]
#                 _img_name = self.img_name_pool[key][index]
#                 anco_sample = {'image': _img, 'label': _target, 'img_name': _img_name, 'dc': domain_code}
#                 if self.transform is not None:
#                     anco_sample = self.transform(anco_sample)
#                 sample.append(anco_sample)
#         else:
#             sample = []
#             for key in self.image_pool:
#                 domain_code = list(self.image_pool.keys()).index(key)
#                 _img = self.image_pool[key][index]
#                 _target = self.label_pool[key][index]
#                 _img_name = self.img_name_pool[key][index]
#                 anco_sample = {'image': _img, 'label': _target, 'img_name': _img_name, 'dc': domain_code}
#                 if self.transform is not None:
#                     anco_sample = self.transform(anco_sample)
#                 sample=anco_sample
#         return sample

def _read_img_into_memory(self):
    img_num = len(self.image_list)
    for index in range(img_num):
        basename = os.path.basename(self.image_list[index]['image'])
        Flag = "NULL"
        if basename[0:2] in self.flags_DGS:
            Flag = 'DGS'
        elif basename[0] in self.flags_REF:
            Flag = 'REF'
        elif basename[0] in self.flags_RIM:
            Flag = 'RIM'
        elif basename[0] in self.flags_REF_val:
            Flag = 'REF_val'
        else:
            print("[ERROR:] Unknown dataset!")
            return 0
        if self.splitid[0] == '4':
            # self.image_pool[Flag].append(Image.open(self.image_list[index]['image']).convert('RGB').resize((256, 256), Image.LANCZOS))
            self.image_pool[Flag].append(Image.open(self.image_list[index]['image']).convert('RGB').crop((144, 144, 144+512, 144+512)).resize((256, 256), Image.LANCZOS))
            _target = np.asarray(Image.open(self.image_list[index]['label']).convert('L'))
            _target = _target[144:144+512, 144:144+512]
            _target = Image.fromarray(_target)
        else:
            self.image_pool[Flag].append(
                Image.open(self.image_list[index]['image']).convert('RGB').resize((256, 256), Image.LANCZOS))
            # self.image_pool[Flag].append(Image.open(self.image_list[index]['image']).convert('RGB'))
            _target = Image.open(self.image_list[index]['label'])

        if _target.mode is 'RGB':
            _target = _target.convert('L')
        if self.state != 'prediction':
            _target = _target.resize((256, 256))
        # print(_target.size)
        # print(_target.mode)
        self.label_pool[Flag].append(_target)
        # if self.split[0:4] in 'test':
        _img_name = self.image_list[index]['image'].split('/')[-1]
        self.img_name_pool[Flag].append(_img_name)


if __name__ == '__main__':
    import custom_transforms as tr
    from utils import decode_segmap
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt

    composed_transforms_tr = transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.RandomSized(512),
        tr.RandomRotate(15),
        tr.ToTensor()])

    voc_train = FundusSegmentation(split='train1',
                                   transform=composed_transforms_tr)

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=2)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = tmp
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0]).astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

            break
    plt.show(block=True)