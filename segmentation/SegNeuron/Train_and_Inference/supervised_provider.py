from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import os
import sys
import random
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.augmentation import SimpleAugment as Filp
from utils.consistency_aug_perturbations import Intensity
from utils.consistency_aug_perturbations import GaussBlur
from utils.consistency_aug_perturbations import GaussNoise
from utils.consistency_aug_perturbations_sup import Cutout
from utils.augmentation import ElasticAugment as Elastic
import imageio
from utils.seg_util import mknhood3d
from utils.aff_util import seg_to_affgraph


class Train(Dataset):
    def __init__(self, cfg):
        super(Train, self).__init__()
        self.cfg = cfg
        self.model_type = cfg.MODEL.model_type

        # split training data
        self.start_slice = cfg.DATA.start_slice
        self.end_slice = cfg.DATA.end_slice
        # augmentation
        self.simple_aug = Filp()
        self.min_noise_std = cfg.DATA.min_noise_std
        self.max_noise_std = cfg.DATA.max_noise_std
        self.min_kernel_size = cfg.DATA.min_kernel_size
        self.max_kernel_size = cfg.DATA.max_kernel_size
        self.min_sigma = cfg.DATA.min_sigma
        self.max_sigma = cfg.DATA.max_sigma
        self.perturbations_init()
        self.crop_from_origin = [20, 128, 128]

        self.dataset = []
        self.labels = []
        # dataset_list = ['J0126-sbem', 'Kasthuri-atum', 'Hemi-brain-fib', 'CREMI-sstem', 'AxonEM[M]-sstem',
        #                 'AxonEM[H]-atum', 'Mira-adwt', 'Fib-25-fib', 'Mira-scn', 'Mira-fish']
        
        # dataset_list = ['AxonEM[H]-atum']
        print(cfg.DATA)

        dataset_list = ['train500']
        # dataset_list = ['train']

        for sub_path in dataset_list:
            self.folder_name = os.path.join(cfg.DATA.data_folder, sub_path)
            file_num = len(os.listdir(self.folder_name)) // 2
            train_datasets = ['%d.tif' % i for i in range(file_num)]
            train_labels = ['%d_MaskIns.tif' % i for i in range(file_num)]

            for k in range(len(train_datasets)):
                print('load ' + self.folder_name + train_datasets[k] + ' ...')
                data = imageio.volread(os.path.join(self.folder_name, train_datasets[k]))
                self.dataset.append(data[:])
                label = imageio.volread(os.path.join(self.folder_name, train_labels[k]))
                self.labels.append(label[:])

        print(len(self.dataset))

    def __getitem__(self, index):
        
        used_data = self.dataset[0]
        used_label = self.labels[0]

        raw_data_shape = used_data.shape

        random_z = random.randint(0, raw_data_shape[0] - self.crop_from_origin[0])
        random_y = random.randint(0, raw_data_shape[1] - self.crop_from_origin[1])
        random_x = random.randint(0, raw_data_shape[2] - self.crop_from_origin[2])

        imgs1 = used_data[random_z:random_z + self.crop_from_origin[0], \
                random_y:random_y + self.crop_from_origin[1], \
                random_x:random_x + self.crop_from_origin[2]].copy()
        lb1 = used_label[random_z:random_z + self.crop_from_origin[0], \
              random_y:random_y + self.crop_from_origin[1], \
              random_x:random_x + self.crop_from_origin[2]].copy()

        imgs1 = imgs1.astype(np.float32) / 255.0

        # normal augumentation
        [imgs1, lb1] = self.simple_aug([imgs1, lb1])
        imgs1, lb1 = self.apply_perturbations(imgs1, lb1)

        lb_affs1 = seg_to_affgraph(lb1, mknhood3d(1), pad='replicate').astype(np.float32)
        bounds1 = np.float32(lb1 != 0)
       
        imgs = imgs1
        lb_affs = lb_affs1
        lb = lb1
        bounds = bounds1

        bmask = np.ones_like(lb)

        # extend dimension
        imgs = imgs[np.newaxis, ...]
        imgs = np.ascontiguousarray(imgs, dtype=np.float32)

        lb_affs = np.ascontiguousarray(lb_affs, dtype=np.float32)

        bounds = bounds[np.newaxis, ...]
        bounds = np.ascontiguousarray(bounds, dtype=np.float32)

        bmask = bmask[np.newaxis, ...]
        bmask = np.ascontiguousarray(bmask, dtype=np.float32)

        return imgs, lb_affs, bounds, bmask

    def perturbations_init(self):
        self.per_intensity = Intensity()
        self.per_gaussnoise = GaussNoise(min_std=self.min_noise_std, max_std=self.max_noise_std, norm_mode='trunc')
        self.per_gaussblur = GaussBlur(min_kernel=self.min_kernel_size, max_kernel=self.max_kernel_size,
                                       min_sigma=self.min_sigma, max_sigma=self.max_sigma)
        self.per_cutout = Cutout(model_type=self.model_type)
        self.per_misalign = Elastic(control_point_spacing=[4, 40, 40], jitter_sigma=[0, 0, 0], prob_slip=0.2,
                                    prob_shift=0.2, max_misalign=17, padding=20)
        self.per_elastic = Elastic(control_point_spacing=[4, 40, 40], jitter_sigma=[0, 2, 2], padding=20)

    def apply_perturbations(self, data, mask):
        if random.random() < 0.25:
            data = self.per_intensity(data)
        if random.random() < 0.25:
            data = self.per_gaussnoise(data)
        if random.random() < 0.25:
            data = self.per_gaussblur(data)
        if random.random() < 0.25:
            data = self.per_cutout(data)
        if random.random() < 0.25:
            data, mask = self.per_elastic(data, mask)

        return data, mask

    def __len__(self):
        return int(sys.maxsize)


class Provider(object):
    def __init__(self, stage, cfg):
        self.stage = stage
        if self.stage == 'train':
            self.data = Train(cfg)
            self.batch_size = cfg.TRAIN.batch_size
            self.num_workers = cfg.TRAIN.num_workers
        elif self.stage == 'valid':
            pass
        else:
            raise AttributeError('Stage must be train/valid')
        self.is_cuda = cfg.TRAIN.if_cuda
        self.data_iter = None
        self.iteration = 0
        self.epoch = 1

    def __len__(self):
        return self.data.num_per_epoch

    def build(self):
        if self.stage == 'train':
            self.data_iter = iter(
                DataLoader(dataset=self.data, batch_size=self.batch_size, num_workers=self.num_workers,
                           shuffle=False, drop_last=False, pin_memory=True))
        else:
            self.data_iter = iter(DataLoader(dataset=self.data, batch_size=1, num_workers=0,
                                             shuffle=False, drop_last=False, pin_memory=True))

    def next(self):
        if self.data_iter is None:
            self.build()
        try:
            batch = next(self.data_iter)
            self.iteration += 1
            if self.is_cuda:
                batch[0] = batch[0].cuda()
                batch[1] = batch[1].cuda()
                batch[2] = batch[2].cuda()
                batch[3] = batch[3].cuda()
            return batch
        except StopIteration:
            self.epoch += 1
            self.build()
            self.iteration += 1
            batch = next(self.data_iter)
            if self.is_cuda:
                batch[0] = batch[0].cuda()
                batch[1] = batch[1].cuda()
                batch[2] = batch[2].cuda()
                batch[3] = batch[3].cuda()
            return batch


def FDA_source_to_target_np(src_img, trg_img, L=0.1):
    # exchange magnitude
    # input: src_img, trg_img

    src_img_np = src_img  # .cpu().numpy()
    trg_img_np = trg_img  # .cpu().numpy()

    # get fft of both source and target
    fft_src_np = np.fft.fft2(src_img_np, axes=(-2, -1))
    fft_trg_np = np.fft.fft2(trg_img_np, axes=(-2, -1))

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
    amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np(amp_src, amp_trg, L=L)

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp(1j * pha_src)

    # get the mutated image
    src_in_trg = np.fft.ifft2(fft_src_, axes=(-2, -1))
    src_in_trg = np.real(src_in_trg)

    return src_in_trg


def low_freq_mutate_np(amp_src, amp_trg, L=0.1):
    a_src = np.fft.fftshift(amp_src, axes=(-2, -1))
    a_trg = np.fft.fftshift(amp_trg, axes=(-2, -1))

    h, w = a_src.shape
    b = (np.floor(np.amin((h, w)) * L)).astype(int)
    c_h = np.floor(h / 2.0).astype(int)
    c_w = np.floor(w / 2.0).astype(int)

    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1

    a_src[h1:h2, w1:w2] = a_trg[h1:h2, w1:w2]
    a_src = np.fft.ifftshift(a_src, axes=(-2, -1))
    return a_src


def FDA_source_to_target_np_3D(src_img, trg_img, L=0.01):
    scr_out = []
    for i in range(src_img.shape[0]):
        scr_out.append(FDA_source_to_target_np(src_img[i], trg_img[i], L))
    return np.array(scr_out)
