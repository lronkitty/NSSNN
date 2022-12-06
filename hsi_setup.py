from math import inf
import torch
import torch.optim as optim
import models
import scipy.io as io
import os
import argparse
from tqdm import tqdm
from os.path import join
from utility import *
from utility import dataloaders_hsi_test
from utility.ssim import SSIMLoss
from indexes import MSIQA
import numpy as np

def resize_ahead(inputs):
    inputs = inputs.cpu().numpy()
    resize_from = (inputs.shape[-3],inputs.shape[-2],inputs.shape[-1])
    resize_to   = (inputs.shape[-3],inputs.shape[-2]//(-8)*(-8),inputs.shape[-1]//(-8)*(-8))
    new_inputs  = np.empty(resize_to)
    # print(inputs.shape,new_inputs.shape,resize_to,(resize_to[-2],resize_to[-1]))
    for b in range(inputs.shape[-3]):
        new_inputs[b,:,:]  = cv2.resize(inputs[0,b,:,:],(resize_to[-1],resize_to[-2]))
        # print(temp.shape)
    inputs = torch.from_numpy(new_inputs).unsqueeze(0)
    return inputs,resize_from

def resize_back(inputs,resize_from):
    inputs = inputs.cpu().numpy()
    new_inputs  = np.empty(resize_from)
    for b in range(inputs.shape[-3]):
        new_inputs[b,:,:]  = cv2.resize(inputs[0,0,b,:,:],(resize_from[-1],resize_from[-2]))
    inputs = torch.from_numpy(new_inputs).unsqueeze(0).unsqueeze(0)
    return inputs



model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


class MultipleLoss(nn.Module):
    def __init__(self, losses, weight=None):
        super(MultipleLoss, self).__init__()
        self.losses = nn.ModuleList(losses)
        self.weight = weight or [1/len(self.losses)] * len(self.losses)
    
    def forward(self, predict, target):
        total_loss = 0
        for weight, loss in zip(self.weight, self.losses):
            total_loss += loss(predict, target) * weight
        return total_loss

    def extra_repr(self):
        return 'weight={}'.format(self.weight)


def train_options(parser):
    def _parse_str_args(args):
        str_args = args.split(',')
        parsed_args = []
        for str_arg in str_args:
            arg = int(str_arg)
            if arg >= 0:
                parsed_args.append(arg)
        return parsed_args    
    parser.add_argument('--prefix', '-p', type=str, default='sru3d_nobn_test',
                        help='prefix')
    parser.add_argument('--arch', '-a', metavar='ARCH',
                        default='sru_ccnet_2d_05_2',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names))
    parser.add_argument('--batchSize', '-b', type=int,
                        default=1, help='training batch size. default=16')         
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate. default=1e-3.')
    parser.add_argument('--wd', type=float, default=0,
                        help='weight decay. default=0')
    parser.add_argument('--loss', type=str, default='l2',
                        help='which loss to choose.', choices=['l1', 'l2', 'smooth_l1', 'ssim', 'l2_ssim'])
    parser.add_argument('--init', type=str, default='kn',
                        help='which init scheme to choose.', choices=['kn', 'ku', 'xn', 'xu', 'edsr'])
    parser.add_argument('--no-cuda', action='store_true', help='disable cuda?')
    parser.add_argument('--no-log', action='store_true',
                        help='disable logger?')
    parser.add_argument('--threads', type=int, default=8,
                        help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=2018,
                        help='random seed to use. default=2018')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--no-ropt', '-nro', action='store_true',
                            help='not resume optimizer')          
    parser.add_argument('--chop', action='store_true',
                            help='forward chop')                                      
    parser.add_argument('--resumePath', '-rp', type=str,
                        default=None, help='checkpoint to use.')
    parser.add_argument('--dataroot', '-d', type=str,
                        default='./datasets/ICVL64_31.db', help='data root')
    parser.add_argument('--clip', type=float, default=1e6)
    parser.add_argument('--gpu-ids', type=str, default='1,3', help='gpu ids')

    parser.add_argument('--testroot', '-tr', type=str,default= '/nas_data/xiongfc/CVPR2022/ICVL/test')
    parser.add_argument('--gtroot', '-gr', type=str,default= '/nas_data/xiongfc/CVPR2022/ICVL/test/test_crop/')
    parser.add_argument('--resetepoch', type=int,default=-1)
    
    parser.add_argument('--output_file_name','-ofn', type=str,default='tmp')
    parser.add_argument('--output_fold','-ofd', type=str,default='results/tmp/')
    parser.add_argument('--save','-s', type=str,default='none',choices=['none','target','all','output'])
    opt = parser.parse_args()
    opt.gpu_ids = _parse_str_args(opt.gpu_ids)
    return opt


def make_dataset(opt, train_transform, target_transform, common_transform, batch_size=None, repeat=1):
    dataset = LMDBDataset(opt.dataroot, repeat=repeat)
    # dataset.length -= 1000
    # dataset.length = size or dataset.length

    """Split patches dataset into training, validation parts"""
    dataset = TransformDataset(dataset, common_transform)

    train_dataset = ImageTransformDataset(dataset, train_transform, target_transform)

    #train_loader = DataLoader(train_dataset,
    #                          batch_size=batch_size or opt.batchSize, shuffle=True,
    #                          num_workers=opt.threads, pin_memory=not opt.no_cuda, worker_init_fn=worker_init_fn)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size or opt.batchSize, shuffle=True,
                              num_workers=opt.threads, pin_memory=not opt.no_cuda, worker_init_fn=worker_init_fn)
    return train_loader


class Engine(object):
    def __init__(self, opt):
        self.prefix = opt.prefix
        self.opt = opt
        self.net = None
        self.optimizer = None
        self.criterion = None
        self.basedir = None
        self.iteration = None
        self.epoch = None
        self.best_psnr = None
        self.best_loss = None
        self.writer = None

        self.__setup()

    def __setup(self):
        self.basedir = join('checkpoints', self.opt.arch)
        if not os.path.exists(self.basedir):
            os.makedirs(self.basedir)
        os.makedirs(self.opt.output_fold,exist_ok=True)
        self.best_psnr = 0
        self.best_loss = 1e6
        self.epoch = 0  # start from epoch 0 or last checkpoint epoch
        self.iteration = 0
        '''
        cuda_list = str(self.opt.gpu_ids)[1:-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_list
        gpus_list = []
        for gpus in range(len(self.opt.gpu_ids)):
            gpus_list.append(gpus)
        self.opt.gpu_ids = gpus_list
        '''
        cuda = not self.opt.no_cuda
        self.device = 'cuda:'+str(self.opt.gpu_ids[0]) if cuda else 'cpu'
        torch.cuda.set_device('cuda:{}'.format(self.opt.gpu_ids[0]))
        print('Cuda Acess: %d' % cuda)
        if cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")

        torch.manual_seed(self.opt.seed)
        if cuda:
            torch.cuda.manual_seed(self.opt.seed)

        """Model"""
        print("=> creating model '{}'".format(self.opt.arch))
        with torch.cuda.device(self.opt.gpu_ids[0]):
            self.net = models.__dict__[self.opt.arch]()
        # initialize parameters
        
        init_params(self.net, init_type=self.opt.init) # disable for default initialization

        if len(self.opt.gpu_ids) > 1:
            from models.sync_batchnorm import DataParallelWithCallback
            self.net = DataParallelWithCallback(self.net, device_ids=self.opt.gpu_ids)
        
        if self.opt.loss == 'l2':
            self.criterion = nn.MSELoss()
        if self.opt.loss == 'l1':
            self.criterion = nn.L1Loss()
        if self.opt.loss == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss()
        if self.opt.loss == 'ssim':
            self.criterion = SSIMLoss(data_range=1, channel=31)
        if self.opt.loss == 'l2_ssim':
            self.criterion = MultipleLoss([nn.MSELoss(), SSIMLoss(data_range=1, channel=31)], weight=[1, 2.5e-3])
        
        print(self.criterion)

        if cuda:
            self.net.to(self.device)
            self.criterion = self.criterion.to(self.device)

        """Logger Setup"""
        log = not self.opt.no_log
        if log:
            self.writer = get_summary_writer(os.path.join(self.basedir, 'logs'), self.opt.prefix)

        """Optimization Setup"""
        self.optimizer = optim.Adam(
            self.net.parameters(), lr=self.opt.lr, weight_decay=self.opt.wd, amsgrad=False)

        """Resume previous model"""
        if self.opt.resume:
            # Load checkpoint.
            self.load(self.opt.resumePath, not self.opt.no_ropt)
        else:
            print('==> Building model..')
            print(self.net)

    def forward(self, inputs):        
        if self.opt.chop:            
            output = self.forward_chop(inputs)
        else:
            output = self.net(inputs)
        
        return output

    def forward_chop(self, x, base=16):        
        n, c, b, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        
        shave_h = np.ceil(h_half / base) * base - h_half
        shave_w = np.ceil(w_half / base) * base - w_half

        shave_h = shave_h if shave_h >= 10 else shave_h + base
        shave_w = shave_w if shave_w >= 10 else shave_w + base

        h_size, w_size = int(h_half + shave_h), int(w_half + shave_w)        
        
        inputs = [
            x[..., 0:h_size, 0:w_size],
            x[..., 0:h_size, (w - w_size):w],
            x[..., (h - h_size):h, 0:w_size],
            x[..., (h - h_size):h, (w - w_size):w]
        ]

        outputs = [self.net(input_i) for input_i in inputs]

        output = torch.zeros_like(x)
        output_w = torch.zeros_like(x)
        
        output[..., 0:h_half, 0:w_half] += outputs[0][..., 0:h_half, 0:w_half]
        output_w[..., 0:h_half, 0:w_half] += 1
        output[..., 0:h_half, w_half:w] += outputs[1][..., 0:h_half, (w_size - w + w_half):w_size]
        output_w[..., 0:h_half, w_half:w] += 1
        output[..., h_half:h, 0:w_half] += outputs[2][..., (h_size - h + h_half):h_size, 0:w_half]
        output_w[..., h_half:h, 0:w_half] += 1
        output[..., h_half:h, w_half:w] += outputs[3][..., (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]
        output_w[..., h_half:h, w_half:w] += 1
        
        output /= output_w

        return output

    def __step(self, train, inputs, targets):        
        if train:
            self.optimizer.zero_grad()
        loss_data = 0
        total_norm = None
        if self.get_net().bandwise:
            O = []
            for time, (i, t) in enumerate(zip(inputs.split(1, 1), targets.split(1, 1))):
                o = self.net(i)
                O.append(o)
                loss = self.criterion(o, t)
                if train:
                    loss.backward()
                loss_data += loss.item()
            outputs = torch.cat(O, dim=1)
        else:
            outputs = self.net(inputs)
            # outputs = torch.clamp(self.net(inputs), 0, 1)
            # loss = self.criterion(outputs, targets)
            
            # if outputs.ndimension() == 5:
            #     loss = self.criterion(outputs[:,0,...], torch.clamp(targets[:,0,...], 0, 1))
            # else:
            #     loss = self.criterion(outputs, torch.clamp(targets, 0, 1))
            
            loss = self.criterion(outputs, targets)
            
            if train:
                loss.backward()
            loss_data += loss.item()
        if train:
            total_norm = nn.utils.clip_grad_norm_(self.net.parameters(), self.opt.clip)
            self.optimizer.step()

        return outputs, loss_data, total_norm

    def load(self, resumePath=None, load_opt=True):
        model_best_path = join(self.basedir, self.prefix, 'model_latest.pth')
        if os.path.exists(model_best_path):
            best_model = torch.load(model_best_path,map_location=torch.device(self.device))

        print('==> Resuming from checkpoint %s..' % resumePath)
        assert os.path.isdir('checkpoints'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(resumePath or model_best_path,map_location=torch.device(self.device))
        #### comment when using memnet
        self.epoch = checkpoint['epoch'] 
        self.iteration = checkpoint['iteration']
        if load_opt:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        ####
        self.get_net().load_state_dict(checkpoint['net'])

    def train(self, train_loader):
        print('\nEpoch: %d' % self.epoch)
        self.net.train()
        train_loss = 0
        #torch.backends.cudnn.enabled = False
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if not self.opt.no_cuda:
                inputs, targets = inputs.to(self.device), targets.to(self.device)            
            outputs, loss_data, total_norm = self.__step(True, inputs, targets)
            # print(targets.min(), targets.max())
            train_loss += loss_data
            avg_loss = train_loss / (batch_idx+1)

            if not self.opt.no_log:
                self.writer.add_scalar(
                    join(self.prefix, 'train_loss'), loss_data, self.iteration)
                self.writer.add_scalar(
                    join(self.prefix, 'train_avg_loss'), avg_loss, self.iteration)

            self.iteration += 1

            progress_bar(batch_idx, len(train_loader), 'AvgLoss: %.4e | Loss: %.4e | Norm: %.4e' 
                         % (avg_loss, loss_data, total_norm))

        self.epoch += 1
        if not self.opt.no_log:
            self.writer.add_scalar(
                join(self.prefix, 'train_loss_epoch'), avg_loss, self.epoch)

    def validate(self, valid_loader, name,size=inf):
        self.net.eval()
        gt_path=self.opt.gtroot
        validate_loss = 0
        total_psnr = 0
        print('\n[i] Eval dataset {}...'.format(name))
        #print(torch.cuda.device_count())
        with torch.no_grad():
            with torch.cuda.device(self.opt.gpu_ids[0]):
                for batch_idx,(inputs,fname) in enumerate(tqdm(valid_loader,disable=True)):
                    torch.cuda.empty_cache()
                    fname=fname[0]
                    targets=dataloaders_hsi_test.get_gt(gt_path,fname)
                    inputs = inputs.unsqueeze(0)
                    targets = targets.unsqueeze(0).unsqueeze(0)
                    inputs, targets=inputs[:,:,:,:,:], targets[:,:,:,:,:]
                    if not self.opt.no_cuda:
                        self.device = 'cuda:'+str(self.opt.gpu_ids[0]) if not self.opt.no_cuda else 'cpu'
                        inputs, targets = inputs.to(self.device), targets.to(self.device)                

                    outputs, loss_data, _ = self.__step(False, inputs, targets)
                    psnr = np.mean(cal_bwpsnr(outputs, targets))
                    #psnr = np.mean(cal_bwpsnr(inputs, targets))
                    validate_loss += loss_data
                    avg_loss = validate_loss / (batch_idx+1)

                    total_psnr += psnr
                    avg_psnr = total_psnr / (batch_idx+1)

                    progress_bar(batch_idx, len(valid_loader) if size > len(valid_loader) else size, 'PSNR: %.4f | Loss: %.4e'
                                % (avg_psnr,avg_loss))
                    torch.cuda.empty_cache()
                    if batch_idx == size-1:
                        return avg_psnr
                return avg_psnr
                
        if not self.opt.no_log:
            self.writer.add_scalar(
                join(self.prefix, name, 'val_loss_epoch'), avg_loss, self.epoch)
            self.writer.add_scalar(
                join(self.prefix, name, 'val_psnr_epoch'), avg_psnr, self.epoch)

        return avg_psnr, avg_loss

    def validate_save(self, valid_loader, name,size=inf,noise=0):
        self.net.eval()
        gt_path=self.opt.gtroot
        validate_loss = 0
        total_psnr = 0
        print('\n[i] Eval dataset {}...'.format(name))
        #print(torch.cuda.device_count())
        with torch.no_grad():
            with torch.cuda.device(self.opt.gpu_ids[0]):
                for batch_idx,(inputs,fname) in enumerate(tqdm(valid_loader,disable=True)):
                    #torch.cuda.empty_cache()
                    fname=fname[0]
                    try:
                        targets=dataloaders_hsi_test.get_gt(gt_path,fname)
                        
                        inputs = inputs.unsqueeze(0)
                        targets = targets.unsqueeze(0).unsqueeze(0)
                    except:
                        print('no gt')
                        inputs = inputs.unsqueeze(0)
                        # targets = targets.unsqueeze(0).unsqueeze(0)
                        targets = inputs[:,:,:,:,:]
                        
                    inputs, targets=inputs[:,:,:,:,:], targets[:,:,:,:,:]
                    if not self.opt.no_cuda:
                        self.device = 'cuda:'+str(self.opt.gpu_ids[0]) if not self.opt.no_cuda else 'cpu'
                        inputs, targets = inputs.to(self.device), targets.to(self.device)                

                    
                    if targets.max()>2:
                        targets= targets/targets.max()
                    outputs, loss_data, _ = self.__step(False, inputs, targets)
                    psnr = np.mean(cal_bwpsnr(outputs, targets))
                    inputs  = np.array(inputs[0,0,:,:,:].permute(1,2,0).cpu())
                    outputs = np.array(outputs[0,0,:,:,:].permute(1,2,0).cpu())
                    targets = np.array(targets[0,0,:,:,:].permute(1,2,0).cpu())
                    isExists=os.path.exists(str(noise)+'dB/')
                    if not isExists:
                        os.makedirs(str(noise)+'dB/')
                    io.savemat(str(noise)+'dB/'+fname+'_inp'+'.mat',{'inputs':inputs})
                    io.savemat(str(noise)+'dB/'+fname+'_out_'+str(round(psnr,2))+'.mat',{'outputs':outputs})
                    io.savemat(str(noise)+'dB/'+fname+'_tar'+'.mat',{'targets':targets})
                    #psnr = np.mean(cal_bwpsnr(inputs, targets))
                    validate_loss += loss_data
                    avg_loss = validate_loss / (batch_idx+1)

                    total_psnr += psnr
                    avg_psnr = total_psnr / (batch_idx+1)

                    progress_bar(batch_idx, len(valid_loader) if size > len(valid_loader) else size, 'PSNR: %.4f | Loss: %.4e'
                                % (avg_psnr,avg_loss))
                    if batch_idx == size-1:
                        return avg_psnr
                return avg_psnr
                
        if not self.opt.no_log:
            self.writer.add_scalar(
                join(self.prefix, name, 'val_loss_epoch'), avg_loss, self.epoch)
            self.writer.add_scalar(
                join(self.prefix, name, 'val_psnr_epoch'), avg_psnr, self.epoch)

        return avg_psnr, avg_loss

    def validate_save_nogt(self, valid_loader, name,size=inf,noise=0):
        self.net.eval()
        gt_path=self.opt.gtroot
        validate_loss = 0
        total_psnr = 0
        print('\n[i] Eval dataset {}...'.format(name))
        #print(torch.cuda.device_count())
        with torch.no_grad():
            with torch.cuda.device(self.opt.gpu_ids[0]):
                for batch_idx,(inputs,fname) in enumerate(tqdm(valid_loader,disable=True)):
                    #torch.cuda.empty_cache()
                    fname=fname[0]
                    # targets=dataloaders_hsi_test.get_gt(gt_path,fname)
                    inputs = inputs.unsqueeze(0)
                    # targets = targets.unsqueeze(0).unsqueeze(0)
                    targets = inputs[:,:,:,:,:]
                    inputs, targets=inputs[:,:,:,:,:], targets[:,:,:,:,:]
                    if not self.opt.no_cuda:
                        self.device = 'cuda:'+str(self.opt.gpu_ids[0]) if not self.opt.no_cuda else 'cpu'
                        inputs, targets = inputs.to(self.device), targets.to(self.device)  
                        # inputs = inputs.to(self.device)          

                    outputs, loss_data, _ = self.__step(False, inputs, targets)
                    psnr = np.mean(cal_bwpsnr(outputs, targets))
                    inputs  = np.array(inputs[0,0,:,:,:].permute(1,2,0).cpu())
                    outputs = np.array(outputs[0,0,:,:,:].permute(1,2,0).cpu())
                    # targets = np.array(targets[0,0,:,:,:].permute(1,2,0).cpu())
                    io.savemat(str(noise)+'_'+fname+'_inp'+'.mat',{'inputs':inputs})
                    io.savemat(str(noise)+'_'+fname+'_out'+'.mat',{noise:outputs})
                    # io.savemat(str(noise)+'dB/'+str(batch_idx)+'_tar'+'.mat',{'targets':targets})
                    #psnr = np.mean(cal_bwpsnr(inputs, targets))
                    validate_loss += loss_data
                    avg_loss = validate_loss / (batch_idx+1)

                    total_psnr += psnr
                    avg_psnr = total_psnr / (batch_idx+1)

                    progress_bar(batch_idx, len(valid_loader) if size > len(valid_loader) else size, 'PSNR: %.4f | Loss: %.4e'
                                % (avg_psnr,avg_loss))
                    if batch_idx == size-1:
                        return avg_psnr
                return avg_psnr
                
        if not self.opt.no_log:
            self.writer.add_scalar(
                join(self.prefix, name, 'val_loss_epoch'), avg_loss, self.epoch)
            self.writer.add_scalar(
                join(self.prefix, name, 'val_psnr_epoch'), avg_psnr, self.epoch)

        return avg_psnr, avg_loss


    def validate_save_cubes(self, valid_loader, name,size=inf,folder=0):
        from utility.read_HSI import read_HSI
        from utility.refold import refold   
        self.net.eval()
        gt_path=self.opt.gtroot
        validate_loss = 0
        total_psnr = 0
        print('\n[i] Eval dataset {}...'.format(name))
        #print(torch.cuda.device_count())
        with torch.no_grad():
            with torch.cuda.device(self.opt.gpu_ids[0]):
                for batch_idx,(inputs,fname) in enumerate(tqdm(valid_loader,disable=True)):
                    #torch.cuda.empty_cache()
                    fname=fname[0]
                    targets=dataloaders_hsi_test.get_gt(gt_path,fname)

                    kernel_size = (31,64,64)
                    kernel_size = (31,inputs.shape[-2],inputs.shape[-1])

                    stride =(5,28,28)
                    # io.savemat(str(folder)+'input_'+fname,{'inputs':inputs[0].permute(1,2,0).numpy()})
                    # print(inputs.shape)
                    col_data,data_shape = read_HSI(inputs[0].numpy(),kernel_size=kernel_size,stride=stride,device='cuda:'+str(self.opt.gpu_ids[0]))
                    #col_data.shape = [n,1,x,y,z] 对col_data进行各种运算，保证shape不变
                    inputs = col_data
                    # inputs = inputs.unsqueeze(0).unsqueeze(0)
                    targets = targets.unsqueeze(0).unsqueeze(0)
                    # targets = inputs[:,:,:,:,:]
                    inputs, targets=inputs[:,:,:,:,:], targets[:,:,:,:,:]
                    if not self.opt.no_cuda:
                        self.device = 'cuda:'+str(self.opt.gpu_ids[0]) if not self.opt.no_cuda else 'cpu'
                        inputs, targets = inputs.to(self.device), targets.to(self.device)  
                        # inputs = inputs.to(self.device)          
                    outputs = torch.empty_like(inputs).to(inputs.device)
                    for batch in range(inputs.shape[0]):
                        print(batch,'/',inputs.shape[0],end='\r')
                        outputs[batch:batch+1,:,:,:,:], loss_data, _ = self.__step(False, inputs[batch:batch+1,:,:,:,:], targets[batch:batch+1,:,:,:,:])
                        torch.cuda.empty_cache()
                    # inputs = refold(col_data,data_shape=data_shape, kernel_size=kernel_size,stride=stride)
                    psnr = np.mean(cal_bwpsnr(outputs, targets))
                    outputs = refold(outputs,data_shape=data_shape, kernel_size=kernel_size,stride=stride,device='cuda:'+str(self.opt.gpu_ids[0])).unsqueeze(0).unsqueeze(0)
                    inputs  = np.array(inputs[0,0,:,:,:].permute(1,2,0).cpu())
                    outputs = np.array(outputs[0,0,:,:,:].permute(1,2,0).cpu())
                    # targets = np.array(targets[0,0,:,:,:].permute(1,2,0).cpu())
                    print(fname,'PSNR:',psnr)
                    io.savemat(str(folder)+'/'+name+'_'+fname+'_'+str(round(psnr,4)),{name:outputs})
                    # io.savemat(str(noise)+'dB/'+str(batch_idx)+'_tar'+'.mat',{'targets':targets})
                    #psnr = np.mean(cal_bwpsnr(inputs, targets))
                    validate_loss += loss_data
                    avg_loss = validate_loss / (batch_idx+1)

                    total_psnr += psnr
                    avg_psnr = total_psnr / (batch_idx+1)

                    progress_bar(batch_idx, len(valid_loader) if size > len(valid_loader) else size, 'PSNR: %.4f | Loss: %.4e'
                                % (avg_psnr,avg_loss))
                    if batch_idx == size-1:
                        return avg_psnr
                return avg_psnr
                
        if not self.opt.no_log:
            self.writer.add_scalar(
                join(self.prefix, name, 'val_loss_epoch'), avg_loss, self.epoch)
            self.writer.add_scalar(
                join(self.prefix, name, 'val_psnr_epoch'), avg_psnr, self.epoch)

        return avg_psnr, avg_loss

    def validate_MSIQA(self, valid_loader, name,size=inf,folder=0,kernel_size=None):
        self.net.eval()
        gt_path=self.opt.gtroot
        validate_loss = 0
        total_psnr = 0
        total_sam  = 0
        total_ssim = 0
        
        flag= False
        filename = str(folder)+'/'+name+'.csv'
        filename_sum = str(folder)+'/'+name+'_msiqa'+'.csv'
        if os.path.exists(filename):
            flag = True
        
        f = open(filename,'a')
        f_sum = open(filename_sum,'a')
        if not flag: 
            f.write("fname,PSNR,SSIM,SAM,timestamp\n")
            f_sum.write("PSNR,SSIM,SAM,timestamp\n")
        output_pattern = "{FNAME},{PSNR},{SSIM},{SAM},{TIMESTAMP}\n"
        output_pattern_sum = "{PSNR},{SSIM},{SAM},{TIMESTAMP}\n"
        # f.write(time.time())
        print('\n[i] Eval dataset {}...'.format(name))
        #print(torch.cuda.device_count())
        with torch.no_grad():
            with torch.cuda.device(self.opt.gpu_ids[0]):
                for batch_idx,(inputs,fname) in enumerate(tqdm(valid_loader,disable=True)):
                    #torch.cuda.empty_cache()
                    fname=fname[0]
                    targets=dataloaders_hsi_test.get_gt(gt_path,fname)
    
                    inputs,resize_from = resize_ahead(inputs)

                    kernel_size = (inputs.shape[-3],inputs.shape[-2],inputs.shape[-1])
                    # if kernel_size is None:
                    #     kernel_size = (inputs.shape[-3],inputs.shape[-2],inputs.shape[-1])
                    # kernel_size = (31,64,64)
                    # kernel_size = (31,inputs.shape[-2],inputs.shape[-1])
                    # kernel_size = (191,64,64)
                    stride =(5,32,32)
                    # print(inputs.shape,targets.shape)
                    col_data,data_shape = read_HSI(inputs[0].numpy(),kernel_size=kernel_size,stride=stride,device='cuda:'+str(self.opt.gpu_ids[0]))    
                    # print(col_data.shape)
                    if col_data.shape[0] ==0:
                        inputs = col_data[0,:,:,:,:].unsqueeze(0)
                    else:
                        inputs = col_data
                    # print(inputs.shape)
                    # inputs = inputs.unsqueeze(0)
                    targets = targets.unsqueeze(0).unsqueeze(0)
                    # if targets.max()>3:
                    # print(inputs.max(),targets.max())
                    # print(inputs.max(),targets.max())
                    targets= targets/targets.max()
                    # print(targets.max(),inputs.max())
                    # print()
                    # else:
                    #     inputs, targets=inputs[:,:,:,:,:], targets[:,:,:,:,:]
                    self.device = 'cuda:'+str(self.opt.gpu_ids[0]) if not self.opt.no_cuda else 'cpu'
                    if not self.opt.no_cuda:
                        
                        inputs, targets = inputs.to(self.device), targets.to(self.device) 
                               
                     

                    outputs = torch.empty_like(inputs).to(inputs.device)
                    for batch in range(inputs.shape[0]):
                        print(batch,'/',inputs.shape[0],end='\r')
                        outputs[batch:batch+1,:,:,:,:], loss_data, _ = self.__step(False, inputs[batch:batch+1,:,:,:,:].to(self.device), inputs[batch:batch+1,:,:,:,:].to(self.device))
                        torch.cuda.empty_cache()
                        # print(outputs.shape)
                    # inputs = refold(col_data,data_shape=data_shape, kernel_size=kernel_size,stride=stride)
                    outputs = refold(outputs,data_shape=data_shape, kernel_size=kernel_size,stride=stride,device='cuda:'+str(self.opt.gpu_ids[0])).unsqueeze(0).unsqueeze(0)
                    # psnr = np.mean(cal_bwpsnr(outputs, targets))
                    outputs = resize_back(outputs,resize_from)
                    psnr, ssim, sam = MSIQA(outputs, targets)
                    f.write(
                        output_pattern.format(
                            FNAME=fname,
                            PSNR=str(psnr),
                            SSIM=str(ssim),
                            SAM=str(sam),
                            TIMESTAMP=time.time()
                        )    
                    )
                    if self.opt.save =='all':
                        inputs  = np.array(inputs[0,0,:,:,:].permute(1,2,0).cpu())
                        targets = np.array(targets[0,0,:,:,:].permute(1,2,0).cpu())
                        outputs = np.array(outputs[0,0,:,:,:].permute(1,2,0).cpu())
                        io.savemat(str(folder)+'/'+'input'+'_'+fname,{'input':inputs})
                        io.savemat(str(folder)+'/'+'target'+'_'+fname,{'target':targets})
                        io.savemat(str(folder)+'/'+name+'_'+str(round(psnr,4))+'_'+fname,{name:outputs})
                    if self.opt.save =='input':
                        inputs  = np.array(inputs[0,0,:,:,:].permute(1,2,0).cpu())
                        # targets = np.array(targets[0,0,:,:,:].permute(1,2,0).cpu())
                        outputs = np.array(outputs[0,0,:,:,:].permute(1,2,0).cpu())
                        io.savemat(str(folder)+'/'+'input'+'_'+fname,{'input':inputs})
                        # io.savemat(str(folder)+'/'+'target'+'_'+fname,{'target':targets})
                        io.savemat(str(folder)+'/'+name+'_'+str(round(psnr,4))+'_'+fname,{name:outputs})
                    if self.opt.save =='target':
                        # inputs  = np.array(inputs[0,0,:,:,:].permute(1,2,0).cpu())
                        targets = np.array(targets[0,0,:,:,:].permute(1,2,0).cpu())
                        outputs = np.array(outputs[0,0,:,:,:].permute(1,2,0).cpu())
                        # io.savemat(str(folder)+'/'+'input'+'_'+fname,{'input':inputs})
                        io.savemat(str(folder)+'/'+'target'+'_'+fname,{'target':targets})
                        io.savemat(str(folder)+'/'+name+'_'+str(round(psnr,4))+'_'+fname,{name:outputs})
                    if self.opt.save =='output':
                        # inputs  = np.array(inputs[0,0,:,:,:].permute(1,2,0).cpu())
                        # targets = np.array(targets[0,0,:,:,:].permute(1,2,0).cpu())
                        outputs = np.array(outputs[0,0,:,:,:].permute(1,2,0).cpu())
                        # io.savemat(str(folder)+'/'+'input'+'_'+fname,{'input':inputs})
                        # io.savemat(str(folder)+'/'+'target'+'_'+fname,{'target':targets})
                        io.savemat(str(folder)+'/'+name+'_'+str(round(psnr,4))+'_'+fname,{name:outputs})
                    # print(fname,'PSNR:',psnr)

                    #psnr = np.mean(cal_bwpsnr(inputs, targets))
                    validate_loss += loss_data
                    avg_loss = validate_loss / (batch_idx+1)

                    total_psnr += psnr
                    total_ssim += ssim
                    total_sam  += sam
                    avg_psnr = total_psnr / (batch_idx+1)
                    avg_ssim = total_ssim / (batch_idx+1)
                    avg_sam  = total_sam / (batch_idx+1)

                    progress_bar(batch_idx, len(valid_loader) if size > len(valid_loader) else size, 'PSNR: %.4f | Loss: %.4e'
                                % (avg_psnr,avg_loss))
                    if batch_idx == size-1:
                        f.close()
                        f_sum.write(
                        output_pattern_sum.format(
                            PSNR=str(avg_psnr),
                            SSIM=str(avg_ssim),
                            SAM=str(avg_sam),
                            TIMESTAMP=time.time()
                        )    )
                        f_sum.close()
                        return avg_psnr,avg_ssim,avg_sam
                f.close()
                f_sum.write(
                output_pattern_sum.format(
                    PSNR=str(avg_psnr),
                    SSIM=str(avg_ssim),
                    SAM=str(avg_sam),
                    TIMESTAMP=time.time()
                )    )
                f_sum.close()
                return avg_psnr,avg_ssim,avg_sam
                
        if not self.opt.no_log:
            self.writer.add_scalar(
                join(self.prefix, name, 'val_loss_epoch'), avg_loss, self.epoch)
            self.writer.add_scalar(
                join(self.prefix, name, 'val_psnr_epoch'), avg_psnr, self.epoch)

        return avg_psnr, avg_loss

    def save_checkpoint(self, model_out_path=None, **kwargs):
        if not model_out_path:
            model_out_path = join(self.basedir, self.prefix, "model_epoch_%d_%d.pth" % (
                self.epoch, self.iteration))

        state = {
            'net': self.get_net().state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'iteration': self.iteration,
        }
        
        state.update(kwargs)

        if not os.path.isdir(join(self.basedir, self.prefix)):
            os.makedirs(join(self.basedir, self.prefix))

        torch.save(state, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    # saving result into disk
    def test_develop(self, test_loader, savedir=None, verbose=True):
        from scipy.io import savemat
        from os.path import basename, exists

        def torch2numpy(hsi):
            if self.net.use_2dconv:
                R_hsi = hsi.data[0].cpu().numpy().transpose((1,2,0))
            else:
                R_hsi = hsi.data[0].cpu().numpy()[0,...].transpose((1,2,0))
            return R_hsi    

        self.net.eval()
        test_loss = 0
        total_psnr = 0
        dataset = test_loader.dataset.dataset

        res_arr = np.zeros((len(test_loader), 3))
        input_arr = np.zeros((len(test_loader), 3))

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                if not self.opt.no_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs, loss_data, _ = self.__step(False, inputs, targets)
                
                test_loss += loss_data
                avg_loss = test_loss / (batch_idx+1)
                
                res_arr[batch_idx, :] = MSIQA(outputs, targets)
                input_arr[batch_idx, :] = MSIQA(inputs, targets)

                """Visualization"""
                # Visualize3D(inputs.data[0].cpu().numpy())
                # Visualize3D(outputs.data[0].cpu().numpy())

                psnr = res_arr[batch_idx, 0]
                ssim = res_arr[batch_idx, 1]
                if verbose:
                    print(batch_idx, psnr, ssim)

                if savedir:
                    filedir = join(savedir, basename(dataset.filenames[batch_idx]).split('.')[0])  
                    outpath = join(filedir, '{}.mat'.format(self.opt.arch))

                    if not exists(filedir):
                        os.mkdir(filedir)

                    if not exists(outpath):
                        savemat(outpath, {'R_hsi': torch2numpy(outputs)})
                        
        return res_arr, input_arr

    def test_real(self, test_loader, savedir=None):
        """Warning: this code is not compatible with bandwise flag"""
        from scipy.io import savemat
        from os.path import basename
        self.net.eval()
        dataset = test_loader.dataset.dataset

        with torch.no_grad():
            for batch_idx, inputs in enumerate(test_loader):
                if not self.opt.no_cuda:
                    inputs = inputs.cuda()      

                outputs = self.forward(inputs)

                """Visualization"""                
                input_np = inputs[0].cpu().numpy()
                output_np = outputs[0].cpu().numpy()

                display = np.concatenate([input_np, output_np], axis=-1)
                
                Visualize3D(display)
                # Visualize3D(outputs[0].cpu().numpy())
                # Visualize3D((outputs-inputs).data[0].cpu().numpy())
                
                if savedir:
                    R_hsi = outputs.data[0].cpu().numpy()[0,...].transpose((1,2,0))     
                    savepath = join(savedir, basename(dataset.filenames[batch_idx]).split('.')[0], self.opt.arch + '.mat')
                    savemat(savepath, {'R_hsi': R_hsi})
        
        return outputs

    def get_net(self):
        if len(self.opt.gpu_ids) > 1:
            return self.net.module
        else:
            return self.net           
