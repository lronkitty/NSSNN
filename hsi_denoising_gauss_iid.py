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
        description='Hyperspectral Image Denoising (Gaussian Noise)')
    opt = train_options(parser)
    print(opt)
    
    print(torch.cuda.device_count())
    """Setup Engine"""
    engine = Engine(opt)

    """Dataset Setting"""
    HSI2Tensor = partial(HSI2Tensor, use_2dconv=engine.get_net().use_2dconv)

    common_transform_1 = lambda x: x

    common_transform_2 = Compose([
        partial(rand_crop, cropx=32, cropy=32),
    ])

    target_transform = HSI2Tensor()
    train_transform_0 = Compose([
        AddNoise(50),
        HSI2Tensor()
    ])
    train_transform_1 = Compose([
        AddNoise(10),
        HSI2Tensor()
    ])
    train_transform_2 = Compose([
        AddNoise(30),
        HSI2Tensor()
    ])
    train_transform_3 = Compose([
        AddNoise(50),
        HSI2Tensor()
    ])
    train_transform_4 = Compose([
        AddNoise(70),
        HSI2Tensor()
    ])
    train_transform_5 = Compose([
        AddNoise(90),
        HSI2Tensor()
    ])
    train_transform_6 = Compose([
        AddNoiseList((10,30,50,70,90)),
        HSI2Tensor()
    ])
    '''
    train_transform_1 = Compose([
        AddNoise(50),
        HSI2Tensor()
    ])
    
    train_transform_2 = Compose([
        AddNoiseBlind([10, 30, 50, 70]),
        HSI2Tensor()
    ])
    '''
    print('==> Preparing data..')
    
    icvl_64_31_TL_0 = make_dataset(
        opt, train_transform_0,
        target_transform, common_transform_1,opt.batchSize )
    icvl_64_31_TL_1 = make_dataset(
        opt, train_transform_1,
        target_transform, common_transform_1, opt.batchSize)
    icvl_64_31_TL_2 = make_dataset(
        opt, train_transform_2,
        target_transform, common_transform_1,opt.batchSize)
    icvl_64_31_TL_3 = make_dataset(
        opt, train_transform_3,
        target_transform, common_transform_1, opt.batchSize)
    # icvl_64_31_TL_4 = make_dataset(
    #     opt, train_transform_4,
        # target_transform, common_transform_2, opt.batchSize*4)
    icvl_64_31_TL_4 = make_dataset(
        opt, train_transform_4,
        target_transform, common_transform_1, opt.batchSize)
        
    icvl_64_31_TL_5 = make_dataset(
        opt, train_transform_5,
        target_transform, common_transform_1, opt.batchSize)
        
    icvl_64_31_TL_6 = make_dataset(
        opt, train_transform_6,
        target_transform, common_transform_1, opt.batchSize)
    '''
    icvl_64_31_TL_2 = make_dataset(
        opt, train_transform_2,
        target_transform, common_transform_2, 64)
    '''
    """Test-Dev"""

    ###modified###
    basefolder = opt.testroot
    mat_names = ['icvl_dynamic_512_50','icvl_dynamic_512_70','icvl_dynamic_512_90']
    #mat_names = ['icvl_512_30', 'icvl_512_50']
    mat_loaders = []
    for noise in (50,70,90):
        test_path = os.path.join(basefolder, str(noise)+'/')
        #print('noise:   ',noise,end='')
        mat_loaders.append(dataloaders_hsi_test.get_dataloaders([test_path],verbose=True,grey=False)['test'])
    ###modified###

    #print(icvl_64_31_TL_0.__len__())
    max=30*7
    if icvl_64_31_TL_0.__len__()*opt.batchSize > 2000:
        max_epoch = max//2
        if_100 = 1
        if_eval = 1
        epoch_per_save = 1
        testsize = 10
    else:
        max_epoch = max
        if_100 = 0
        if_eval = 1
        epoch_per_save = 10
        testsize = 5
    print('max_epoch: ',max_epoch)
    """Main loop"""
    base_lr = opt.lr   
    if_val_any = 1
    if opt.resetepoch != -1:
        engine.epoch = opt.resetepoch
    while engine.epoch < max_epoch:
        if if_100:
            epoch = engine.epoch * 2
        else:
            epoch = engine.epoch
        display_learning_rate(engine.optimizer)
        np.random.seed() # reset seed per epoch, otherwise the noise will be added with a specific pattern
        if epoch == 0:
            adjust_learning_rate(engine.optimizer, opt.lr) 
        elif epoch == 10:
            adjust_learning_rate(engine.optimizer, base_lr*0.1)
        elif epoch == 20:
            adjust_learning_rate(engine.optimizer, base_lr*0.01)
        elif epoch % 30 == 0 and epoch >29 and epoch < max:
            adjust_learning_rate(engine.optimizer, base_lr*0.1)
        elif epoch % 30 == 14 and epoch >29 and epoch < max:
            adjust_learning_rate(engine.optimizer, base_lr*0.01)
        elif epoch == max:
            adjust_learning_rate(engine.optimizer, base_lr*0.001)
        '''
        elif engine.epoch % 30 == 1 and engine.epoch != 1:
            adjust_learning_rate(engine.optimizer, base_lr)

        elif engine.epoch % 30 == 0 and engine.epoch != 0:
            adjust_learning_rate(engine.optimizer, base_lr*0.01)
        '''
        #print(if_100)
        if epoch < 30:
            #engine.validate(mat_loaders[0], 'icvl-validate-50')
            print("Training with unbindwise noise 50dB")
            engine.train(icvl_64_31_TL_0)
            if if_val_any:
                engine.validate(mat_loaders[0], 'icvl-validate-50',testsize)###modified###
                if if_eval:
                    engine.validate(mat_loaders[1], 'icvl-validate-70',testsize)###modified###
                    engine.validate(mat_loaders[2], 'icvl-validate-90',testsize)###modified###
            #engine.validate(mat_loaders[1], 'icvl-validate-50')
        elif epoch < 60:
            print("Training with 10dB")
            engine.train(icvl_64_31_TL_1)
            if if_val_any:
                engine.validate(mat_loaders[0], 'icvl-validate-50',testsize)###modified###
                if if_eval:
                    engine.validate(mat_loaders[1], 'icvl-validate-70',testsize)###modified###
                    engine.validate(mat_loaders[2], 'icvl-validate-90',testsize)###modified###
            #engine.validate(mat_loaders[0], 'icvl-validate-50')
            #engine.validate(mat_loaders[0], 'icvl-validate-30')
            #engine.validate(mat_loaders[1], 'icvl-validate-50')
        elif epoch < 90:
            print("Training with 30dB")
            engine.train(icvl_64_31_TL_2)
            if if_val_any:
                engine.validate(mat_loaders[0], 'icvl-validate-50',testsize)###modified###
                engine.validate(mat_loaders[1], 'icvl-validate-70',testsize)###modified###
                if if_eval:
                    engine.validate(mat_loaders[2], 'icvl-validate-90',testsize)###modified###
        elif epoch < 120:
            print("Training with 50dB")
            engine.train(icvl_64_31_TL_3)
            if if_val_any:
                engine.validate(mat_loaders[0], 'icvl-validate-50',testsize)###modified###
                engine.validate(mat_loaders[1], 'icvl-validate-70',testsize)###modified###
                engine.validate(mat_loaders[2], 'icvl-validate-90',testsize)###modified###
        elif epoch < 150:
            print("Training with 70dB")
            engine.train(icvl_64_31_TL_4)
            if if_val_any:
                engine.validate(mat_loaders[0], 'icvl-validate-50',testsize)###modified###
                engine.validate(mat_loaders[1], 'icvl-validate-70',testsize)###modified###
                engine.validate(mat_loaders[2], 'icvl-validate-90',testsize)###modified###
        elif epoch < 180:
            print("Training with 90dB")
            engine.train(icvl_64_31_TL_5)
            if if_val_any:
                engine.validate(mat_loaders[0], 'icvl-validate-50',testsize)###modified###
                engine.validate(mat_loaders[1], 'icvl-validate-70',testsize)###modified###
                engine.validate(mat_loaders[2], 'icvl-validate-90',testsize)###modified###
        else:
            print("Training with random noise")
            engine.train(icvl_64_31_TL_6)
            if engine.epoch == max_epoch and engine.epoch == 15*7:
                testsize = 50
                MSIQAs = []
                MSIQAs.append(engine.validate_MSIQA(mat_loaders[0], 'icvl-validate-50',folder='nssnn_iid'))
                MSIQAs.append(engine.validate_MSIQA(mat_loaders[0], 'icvl-validate-70',folder='nssnn_iid'))
                MSIQAs.append(engine.validate_MSIQA(mat_loaders[0], 'icvl-validate-90',folder='nssnn_iid'))
                print("      PSNR       SSIM      SAM")
                print("50dB: %.4f    %.4f    %.4f"%( MSIQAs[0][0],MSIQAs[0][1],MSIQAs[0][2]))
                print("70dB: %.4f    %.4f    %.4f"%( MSIQAs[1][0],MSIQAs[1][1],MSIQAs[1][2]))
                print("90dB: %.4f    %.4f    %.4f"%( MSIQAs[2][0],MSIQAs[2][1],MSIQAs[2][2]))
            else:
                if if_val_any:
                    engine.validate(mat_loaders[0], 'icvl-validate-50',testsize)###modified###
                    engine.validate(mat_loaders[1], 'icvl-validate-70',testsize)###modified###
                    engine.validate(mat_loaders[2], 'icvl-validate-90',testsize)###modified###
        
        print('\nLatest Result Saving...')
        model_latest_path = os.path.join(engine.basedir, engine.prefix, 'model_latest.pth')
        engine.save_checkpoint(
            model_out_path=model_latest_path
        )

        
        if engine.epoch % epoch_per_save == 0:###modified###
            engine.save_checkpoint()
