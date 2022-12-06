import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from utility import dataloaders_hsi_test
from utility import *
from hsi_setup import Engine, train_options
import models
from indexes import MSIQA

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
    ###modified###
    MSIQAs = []
    basefolder = opt.testroot
    psnrs = []
    test_path = os.path.join(basefolder)
    test = dataloaders_hsi_test.get_dataloaders([test_path],verbose=True,grey=False)
    MSIQAs.append(engine.validate_MSIQA(test['test'],folder=opt.output_fold,name=opt.output_file_name))
    # res_arr, input_arr = engine.test_develop(mat_loader, savedir=resdir, verbose=True)
    # print(res_arr.mean(axis=0))
    # print(opt.output_file_name,opt.output_fold)
    for MSIQA in MSIQAs: 
        for index in MSIQA:
            print("%.4f"%(index))