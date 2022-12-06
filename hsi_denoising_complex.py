import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse

from utility import *
from hsi_setup import Engine, train_options, make_dataset
from utility import dataloaders_hsi_test ###modified###

if __name__ == '__main__':
    """Training settings"""
    parser = argparse.ArgumentParser(
        description='Hyperspectral Image Denoising (Complex noise)')
    opt = train_options(parser)
    print(opt)
    
    print(torch.cuda.device_count())
    """Setup Engine"""
    engine = Engine(opt)

    """Dataset Setting"""
    HSI2Tensor = partial(HSI2Tensor, use_2dconv=engine.get_net().use_2dconv)

    add_noniid_noise = Compose([
        AddNoiseDynamic(95),
        SequentialSelect(
            transforms=[
                lambda x: x,
                AddNoiseImpulse(),
                AddNoiseStripe(),
                AddNoiseDeadline()
            ]
        )
    ])
    common_transform_1 = lambda x: x
    common_transform = Compose([
        partial(rand_crop, cropx=32, cropy=32),
    ])

    target_transform = HSI2Tensor()

    train_transform = Compose([
        add_noniid_noise,
        HSI2Tensor()
    ])

    print('==> Preparing data..')

    icvl_64_31_TL = make_dataset(
        opt, train_transform,
        target_transform, common_transform_1, 16)

    """Test-Dev"""
    basefolder = opt.testroot
    mat_loaders = []
    test_path = os.path.join(basefolder)
        #print('noise:   ',noise,end='')
    mat_loaders.append(dataloaders_hsi_test.get_dataloaders([test_path],verbose=True,grey=False)['test'])       

    base_lr = opt.lr
    base_lr = 1e-4
    epoch_per_save = 10
    if opt.resetepoch != -1:
        engine.epoch = opt.resetepoch
    adjust_learning_rate(engine.optimizer, opt.lr)
    # from epoch 50 to 100
    while engine.epoch < 100:
        display_learning_rate(engine.optimizer)
        np.random.seed()
        if engine.epoch == 85:
            adjust_learning_rate(engine.optimizer, base_lr*0.3)
        
        if engine.epoch == 95:
            adjust_learning_rate(engine.optimizer, base_lr*0.1)

        print("Training with complex")
        engine.train(icvl_64_31_TL)
        if engine.epoch == 100:
            MSIQAs=engine.validate_MSIQA(mat_loaders[0],folder='results/nssrnn/icvl/mix/',name='nssnn_mix',size=50)
            print("%.4f    %.4f    %.4f"%( MSIQAs[0],MSIQAs[1],MSIQAs[2]))

        else:
            MSIQAs=engine.validate_MSIQA(mat_loaders[0],name='nssnn_mix',size=2)
            print("%.4f    %.4f    %.4f"%( MSIQAs[0],MSIQAs[1],MSIQAs[2]))

        
        print('\nLatest Result Saving...')
        model_latest_path = os.path.join(engine.basedir, engine.prefix, 'model_latest.pth')
        engine.save_checkpoint(
            model_out_path=model_latest_path
        )

        
        if engine.epoch % epoch_per_save == 0:###modified###
            engine.save_checkpoint()

    MSIQAs = []       
    for mat_loader in mat_loaders:
        MSIQAs.append(engine.validate_MSIQA(mat_loader,folder=opt.output_fold,name=opt.output_file_name))
    for MSIQA in MSIQAs: 
        for index in MSIQA:
            print("%.4f"%(index))