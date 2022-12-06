import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import argparse

from utility import *
from hsi_setup import Engine, train_options
import models


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


prefix = 'test'

if __name__ == '__main__':
    """Training settings"""
    parser = argparse.ArgumentParser(
        description='Hyperspectral Image Denoising')
    opt = train_options(parser)
    print(opt)

    cuda = not opt.no_cuda
    opt.no_log = True

    """Setup Engine"""
    engine = Engine(opt)

    mat_dataset = MatDataFromFolder('/home/fugym/HDD/fugym/QRNN3D/matlab/Data/icvl_dynamic_512_15')
    
    mat_transform = Compose([
        LoadMatKey(key='img'), # for testing
        lambda x: x[:,:220,:256][None],
        minmax_normalize,
    ])

    mat_dataset = TransformDataset(mat_dataset, mat_transform)

    mat_loader = DataLoader(
                    mat_dataset,
                    batch_size=1, shuffle=False,
                    num_workers=1, pin_memory=cuda
                )
    
    # print(engine.net)
    
    engine.test_real(mat_loader, savedir=None)
