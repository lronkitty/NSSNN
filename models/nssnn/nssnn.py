import torch
import torch.nn as nn
from ..combinations import BasicConv3d,BasicDeConv3d,BasicUpsampleConv3d, Conv3dBNReLU
from .cc import CC_module
act = 'tanh'
import scipy.io as io
class SiSRUconv(nn.Module):
    def __init__(self, in_channels, out_channels, k, s, p,upsample=(1,2,2), bias=False,bn=True,model='unknown'):
        super(SiSRUconv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if model == 'conv':
            self.conv = BasicConv3d(self.in_channels,self.out_channels*4,k, s, p, bias=False,bn=bn)
        elif model == 'deconv':
            self.conv = BasicDeConv3d(self.in_channels,self.out_channels*4,k, s, p, bias=False,bn=bn)
        elif model == 'upsampleconv':
            self.conv = BasicUpsampleConv3d(self.in_channels, self.out_channels*4, k, s, p, upsample=upsample, bn=bn)

    def SRUconvLayer(self,xs_,Ct,Wxs_,fts_,rts_):
        if Ct is None:
            Ct = 1-fts_
        else:
            Ct = fts_*Ct + (1-fts_)*Wxs_
        #ht = rts_*gs_*Ct+ (1-rts_)* xs_
        ht = rts_*Ct+ (1-rts_)* xs_
        return Ct,ht

    def SRUconvGates(self,input):
        gates = self.conv(input)
        Wx,ft,rt,X = gates.split(split_size=self.out_channels,dim=1)
        #X = self.convX(input)
        Wx = Wx.tanh()
        ft = ft.sigmoid()
        rt = rt.sigmoid()
        if act == 'tanh':
            X = X.tanh()
        elif act == 'relu':
            X = X.relu()
        elif act == 'sigmoid':
            X = X.sigmoid()
        elif act == 'none':
            X = X
        return Wx,ft,rt,X

    def forward(self,x,reverse=False):
        Wx,ft,rt,X = self.SRUconvGates(x)
        Wxs = Wx.split(1, 2)
        fts = ft.split(1, 2)
        rts = rt.split(1, 2)
        #rt2s = rt2.split(1,2)
        #xs = torch.repeat_interleave(x, repeats=self.out_channels, dim=1).split(1,2)
        xs = X.split(1,2)
        hts = []
        #print(reverse)
        if not reverse:
            Ct = None
            ht = None
            for time, (xs_,Wxs_,fts_,rts_) in enumerate(zip(xs,Wxs,fts,rts)):
                Ct,ht = self.SRUconvLayer(xs_,Ct,Wxs_,fts_,rts_)
                hts.append(ht)
        else:
            Ct = None
            ht = None
            for time, (xs_,Wxs_,fts_,rts_) in enumerate(zip(reversed(xs), reversed(Wxs), reversed(fts),reversed(rts))):  # split along timestep
                Ct,ht = self.SRUconvLayer(xs_,Ct,Wxs_,fts_,rts_)
                hts.insert(0, ht)
        hts = torch.cat(hts, dim=2)
        
        return hts

class DoSRUconv(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, s=1, p=1, bias=False,bn=True,model='unknown'):
        super(DoSRUconv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if model == 'conv':
            self.conv = BasicConv3d(self.in_channels,self.out_channels*6,k, s, p, bias=bias,bn=bn)
            #self.convX = torch.nn.Conv3d(self.in_channels,self.out_channels,1,1,0)
        elif model == 'deconv':
            self.conv = BasicDeConv3d(self.in_channels,self.out_channels*6,k, s, p, bias=bias,bn=bn)
            #self.convX = torch.nn.ConvTranspose3d(self.in_channels,self.out_channels,1,1,0)
    def SRUconvGates(self,input):
        gates = self.conv(input)
        Wx,ft,ft2,rt,rt2,X = gates.split(split_size=self.out_channels,dim=1)
        #X = self.convX(input)
        Wx = Wx.tanh()
        ft = ft.sigmoid()
        ft2 = ft2.sigmoid()
        rt = rt.sigmoid()
        rt2 = rt2.sigmoid()
        if act == 'tanh':
            X = X.tanh()
        elif act == 'relu':
            X = X.relu()
        elif act == 'sigmoid':
            X = X.sigmoid()
        elif act == 'none':
            X = X
        return Wx,ft,ft2,rt,rt2,X

    def SRUconvLayer(self,xs_,Ct,Wxs_,fts_,rts_):
        if Ct is None:
            Ct = 1-fts_
        else:
            Ct = fts_*Ct + (1-fts_)*Wxs_
        #ht = rts_*gs_*Ct+ (1-rts_)* xs_
        ht = rts_*Ct+ (1-rts_)* xs_
        return Ct,ht

    def forward(self,x):
        Wx,ft,ft2,rt,rt2,X = self.SRUconvGates(x)
        Wxs = Wx.split(1, 2)
        fts = ft.split(1, 2)
        ft2s = ft2.split(1,2)
        rts = rt.split(1, 2)
        rt2s = rt2.split(1,2)
        xs = X.split(1,2)
        Ct = None
        ht = None
        htl = []
        
        for time, (xs_,Wxs_,fts_,rts_) in enumerate(zip(xs,Wxs,fts,rts)):
            Ct,ht = self.SRUconvLayer(xs_,Ct,Wxs_,fts_,rts_)
            htl.append(ht)
        htl = torch.cat(htl, dim=2)
        
        Ct = None
        ht = None
        htr = []
        
        for time, (xs_,Wxs_,fts_,rts_) in enumerate(zip(reversed(xs), reversed(Wxs), reversed(ft2s),reversed(rt2s))):  # split along timestep
            Ct,ht = self.SRUconvLayer(xs_,Ct,Wxs_,fts_,rts_)
            htr.insert(0, ht)
        htr = torch.cat(htr, dim=2)
        #end = torch.cat([htl,htr],dim=2)
        return htr + htl# +X

class SiSRUconv_ccnet(nn.Module):
    def __init__(self, in_channels, out_channels, k, s, p,upsample=(1,2,2), bias=False,bn=True,model='unknown'):
        super(SiSRUconv_ccnet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if model == 'conv':
            self.conv = BasicConv3d(self.in_channels,self.out_channels*4,k, s, p, bias=False,bn=bn)
        elif model == 'deconv':
            self.conv = BasicDeConv3d(self.in_channels,self.out_channels*4,k, s, p, bias=False,bn=bn)
        elif model == 'upsampleconv':
            self.conv = BasicUpsampleConv3d(self.in_channels, self.out_channels*4, k, s, p, upsample=upsample, bn=bn)
        self.cc_model = CC_module(self.out_channels)
        self.conv_before_cc = BasicConv3d(self.out_channels,self.out_channels,k=(1,3,3), s=1, p=(0,1,1), bias=False,bn=bn)
        
    def SRUconvLayer(self,xs_,Ct,Wxs_,fts_,rts_):
        if Ct is None:
            Ct = 1-fts_
        else:
            Ct = fts_*Ct + (1-fts_)*Wxs_
        #ht = rts_*gs_*Ct+ (1-rts_)* xs_
        ht = rts_*Ct+ (1-rts_)* xs_
        return Ct,ht

    def SRUconvGates(self,input):
        gates = self.conv(input)
        Wx,ft,rt,X = gates.split(split_size=self.out_channels,dim=1)
        #X = self.convX(input)
        Wx = Wx.tanh()
        ft = ft.sigmoid()
        rt = rt.sigmoid()
        if act == 'tanh':
            X = X.tanh()
        elif act == 'relu':
            X = X.relu()
        elif act == 'sigmoid':
            X = X.sigmoid()
        elif act == 'none':
            X = X
        return Wx,ft,rt,X

    def forward(self,x,reverse=False):
        Wx,ft,rt,X = self.SRUconvGates(x)
        Wxs = Wx.split(1, 2)
        fts = ft.split(1, 2)
        rts = rt.split(1, 2)
        #rt2s = rt2.split(1,2)
        #xs = torch.repeat_interleave(x, repeats=self.out_channels, dim=1).split(1,2)
        xs = X.split(1,2)
        hts = []
        #print(reverse)
        if not reverse:
            Ct = None
            ht = None
            for time, (xs_,Wxs_,fts_,rts_) in enumerate(zip(xs,Wxs,fts,rts)):
                Ct,ht = self.SRUconvLayer(xs_,Ct,Wxs_,fts_,rts_)
                hts.append(ht)
        else:
            Ct = None
            ht = None
            for time, (xs_,Wxs_,fts_,rts_) in enumerate(zip(reversed(xs), reversed(Wxs), reversed(fts),reversed(rts))):  # split along timestep
                Ct,ht = self.SRUconvLayer(xs_,Ct,Wxs_,fts_,rts_)
                hts.insert(0, ht)
        hts = torch.cat(hts, dim=2)
        #hts = self.conv_before_cc(hts)
        for j in range(2):
            for i in range(hts.shape[2]):
                hts[:,:,i,:,:]= self.cc_model(hts[:,:,i,:,:])
        return hts# +X

class DoSRUconv_ccnet(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, s=1, p=1, bias=False,bn=True,model='unknown'):
        super(DoSRUconv_ccnet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if model == 'conv':
            self.conv = BasicConv3d(self.in_channels,self.out_channels*6,k, s, p, bias=bias,bn=bn)
            #self.convX = torch.nn.Conv3d(self.in_channels,self.out_channels,1,1,0)
        elif model == 'deconv':
            self.conv = BasicDeConv3d(self.in_channels,self.out_channels*6,k, s, p, bias=bias,bn=bn)
            #self.convX = torch.nn.ConvTranspose3d(self.in_channels,self.out_channels,1,1,0)
        self.cc_model = CC_module(self.out_channels)
        self.conv_before_cc = BasicConv3d(self.out_channels,self.out_channels,k=(1,3,3), s=1, p=(0,1,1), bias=False,bn=bn)

    def SRUconvGates(self,input):
        gates = self.conv(input)
        Wx,ft,ft2,rt,rt2,X = gates.split(split_size=self.out_channels,dim=1)
        #X = self.convX(input)
        Wx = Wx.tanh()
        ft = ft.sigmoid()
        ft2 = ft2.sigmoid()
        rt = rt.sigmoid()
        rt2 = rt2.sigmoid()
        if act == 'tanh':
            X = X.tanh()
        elif act == 'relu':
            X = X.relu()
        elif act == 'sigmoid':
            X = X.sigmoid()
        elif act == 'none':
            X = X
        return Wx,ft,ft2,rt,rt2,X

    def SRUconvLayer(self,xs_,Ct,Wxs_,fts_,rts_):
        if Ct is None:
            Ct = 1-fts_
        else:
            Ct = fts_*Ct + (1-fts_)*Wxs_
        #ht = rts_*gs_*Ct+ (1-rts_)* xs_
        ht = rts_*Ct+ (1-rts_)* xs_
        return Ct,ht

    def forward(self,x):
        Wx,ft,ft2,rt,rt2,X = self.SRUconvGates(x)
        Wxs = Wx.split(1, 2)
        fts = ft.split(1, 2)
        ft2s = ft2.split(1,2)
        rts = rt.split(1, 2)
        rt2s = rt2.split(1,2)
        xs = X.split(1,2)
        Ct = None
        ht = None
        htl = []
        
        for time, (xs_,Wxs_,fts_,rts_) in enumerate(zip(xs,Wxs,fts,rts)):
            Ct,ht = self.SRUconvLayer(xs_,Ct,Wxs_,fts_,rts_)
            htl.append(ht)
        htl = torch.cat(htl, dim=2)
        
        Ct = None
        ht = None
        htr = []
        
        for time, (xs_,Wxs_,fts_,rts_) in enumerate(zip(reversed(xs), reversed(Wxs), reversed(ft2s),reversed(rt2s))):  # split along timestep
            Ct,ht = self.SRUconvLayer(xs_,Ct,Wxs_,fts_,rts_)
            htr.insert(0, ht)
        htr = torch.cat(htr, dim=2)
        #end = torch.cat([htl,htr],dim=2)
        out  = htr + htl
        #out = self.conv_before_cc(out)
        for j in range(2):
            for i in range(out.shape[2]):
                out[:,:,i,:,:]= self.cc_model(out[:,:,i,:,:])
        return out# +X

class SiSRUconv_convbn_ccnet(nn.Module):
    def __init__(self, in_channels, out_channels, k, s, p,upsample=(1,2,2), bias=False,bn=True,model='unknown'):
        super(SiSRUconv_convbn_ccnet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if model == 'conv':
            self.conv = BasicConv3d(self.in_channels,self.out_channels*4,k, s, p, bias=False,bn=bn)
        elif model == 'deconv':
            self.conv = BasicDeConv3d(self.in_channels,self.out_channels*4,k, s, p, bias=False,bn=bn)
        elif model == 'upsampleconv':
            self.conv = BasicUpsampleConv3d(self.in_channels, self.out_channels*4, k, s, p, upsample=upsample, bn=bn)
        self.cc_model = CC_module(self.out_channels)
        self.conv_before_cc = BasicConv3d(self.out_channels,self.out_channels,k=(1,3,3), s=1, p=(0,1,1), bias=False,bn=bn)
        
    def SRUconvLayer(self,xs_,Ct,Wxs_,fts_,rts_):
        if Ct is None:
            Ct = 1-fts_
        else:
            Ct = fts_*Ct + (1-fts_)*Wxs_
        #ht = rts_*gs_*Ct+ (1-rts_)* xs_
        ht = rts_*Ct+ (1-rts_)* xs_
        return Ct,ht

    def SRUconvGates(self,input):
        gates = self.conv(input)
        Wx,ft,rt,X = gates.split(split_size=self.out_channels,dim=1)
        #X = self.convX(input)
        Wx = Wx.tanh()
        ft = ft.sigmoid()
        rt = rt.sigmoid()
        if act == 'tanh':
            X = X.tanh()
        elif act == 'relu':
            X = X.relu()
        elif act == 'sigmoid':
            X = X.sigmoid()
        elif act == 'none':
            X = X
        return Wx,ft,rt,X

    def forward(self,x,reverse=False):
        Wx,ft,rt,X = self.SRUconvGates(x)
        Wxs = Wx.split(1, 2)
        fts = ft.split(1, 2)
        rts = rt.split(1, 2)
        #rt2s = rt2.split(1,2)
        #xs = torch.repeat_interleave(x, repeats=self.out_channels, dim=1).split(1,2)
        xs = X.split(1,2)
        hts = []
        #print(reverse)
        if not reverse:
            Ct = None
            ht = None
            for time, (xs_,Wxs_,fts_,rts_) in enumerate(zip(xs,Wxs,fts,rts)):
                Ct,ht = self.SRUconvLayer(xs_,Ct,Wxs_,fts_,rts_)
                hts.append(ht)
        else:
            Ct = None
            ht = None
            for time, (xs_,Wxs_,fts_,rts_) in enumerate(zip(reversed(xs), reversed(Wxs), reversed(fts),reversed(rts))):  # split along timestep
                Ct,ht = self.SRUconvLayer(xs_,Ct,Wxs_,fts_,rts_)
                hts.insert(0, ht)
        hts = torch.cat(hts, dim=2)
        hts = self.conv_before_cc(hts)
        for j in range(2):
            for i in range(hts.shape[2]):
                hts[:,:,i,:,:]= self.cc_model(hts[:,:,i,:,:])
        return hts# +X

class DoSRUconv_convbn_ccnet(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, s=1, p=1, bias=False,bn=True,model='unknown'):
        super(DoSRUconv_convbn_ccnet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if model == 'conv':
            self.conv = BasicConv3d(self.in_channels,self.out_channels*6,k, s, p, bias=bias,bn=bn)
            #self.convX = torch.nn.Conv3d(self.in_channels,self.out_channels,1,1,0)
        elif model == 'deconv':
            self.conv = BasicDeConv3d(self.in_channels,self.out_channels*6,k, s, p, bias=bias,bn=bn)
            #self.convX = torch.nn.ConvTranspose3d(self.in_channels,self.out_channels,1,1,0)
        self.cc_model = CC_module(self.out_channels)
        self.conv_before_cc = BasicConv3d(self.out_channels,self.out_channels,k=(1,3,3), s=1, p=(0,1,1), bias=False,bn=bn)

    def SRUconvGates(self,input):
        gates = self.conv(input)
        Wx,ft,ft2,rt,rt2,X = gates.split(split_size=self.out_channels,dim=1)
        #X = self.convX(input)
        Wx = Wx.tanh()
        ft = ft.sigmoid()
        ft2 = ft2.sigmoid()
        rt = rt.sigmoid()
        rt2 = rt2.sigmoid()
        if act == 'tanh':
            X = X.tanh()
        elif act == 'relu':
            X = X.relu()
        elif act == 'sigmoid':
            X = X.sigmoid()
        elif act == 'none':
            X = X
        return Wx,ft,ft2,rt,rt2,X

    def SRUconvLayer(self,xs_,Ct,Wxs_,fts_,rts_):
        if Ct is None:
            Ct = 1-fts_
        else:
            Ct = fts_*Ct + (1-fts_)*Wxs_
        #ht = rts_*gs_*Ct+ (1-rts_)* xs_
        ht = rts_*Ct+ (1-rts_)* xs_
        return Ct,ht

    def forward(self,x):
        Wx,ft,ft2,rt,rt2,X = self.SRUconvGates(x)
        Wxs = Wx.split(1, 2)
        fts = ft.split(1, 2)
        ft2s = ft2.split(1,2)
        rts = rt.split(1, 2)
        rt2s = rt2.split(1,2)
        xs = X.split(1,2)
        Ct = None
        ht = None
        htl = []
        
        for time, (xs_,Wxs_,fts_,rts_) in enumerate(zip(xs,Wxs,fts,rts)):
            Ct,ht = self.SRUconvLayer(xs_,Ct,Wxs_,fts_,rts_)
            htl.append(ht)
        htl = torch.cat(htl, dim=2)
        
        Ct = None
        ht = None
        htr = []
        
        for time, (xs_,Wxs_,fts_,rts_) in enumerate(zip(reversed(xs), reversed(Wxs), reversed(ft2s),reversed(rt2s))):  # split along timestep
            Ct,ht = self.SRUconvLayer(xs_,Ct,Wxs_,fts_,rts_)
            htr.insert(0, ht)
        htr = torch.cat(htr, dim=2)
        #end = torch.cat([htl,htr],dim=2)
        out  = htr + htl
        out = self.conv_before_cc(out)
        for j in range(2):
            for i in range(out.shape[2]):
                out[:,:,i,:,:]= self.cc_model(out[:,:,i,:,:])
        return out# +X

class SiSRUconv_convbnrelu_ccnet(nn.Module):
    def __init__(self, in_channels, out_channels, k, s, p,upsample=(1,2,2), bias=False,bn=True,model='unknown'):
        super(SiSRUconv_convbnrelu_ccnet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if model == 'conv':
            self.conv = BasicConv3d(self.in_channels,self.out_channels*4,k, s, p, bias=False,bn=bn)
        elif model == 'deconv':
            self.conv = BasicDeConv3d(self.in_channels,self.out_channels*4,k, s, p, bias=False,bn=bn)
        elif model == 'upsampleconv':
            self.conv = BasicUpsampleConv3d(self.in_channels, self.out_channels*4, k, s, p, upsample=upsample, bn=bn)
        self.cc_model = CC_module(self.out_channels)
        self.conv_before_cc = Conv3dBNReLU(self.out_channels,self.out_channels,k=(1,3,3), s=1, p=(0,1,1))
        
    def SRUconvLayer(self,xs_,Ct,Wxs_,fts_,rts_):
        if Ct is None:
            Ct = 1-fts_
        else:
            Ct = fts_*Ct + (1-fts_)*Wxs_
        #ht = rts_*gs_*Ct+ (1-rts_)* xs_
        ht = rts_*Ct+ (1-rts_)* xs_
        return Ct,ht

    def SRUconvGates(self,input):
        gates = self.conv(input)
        Wx,ft,rt,X = gates.split(split_size=self.out_channels,dim=1)
        #X = self.convX(input)
        Wx = Wx.tanh()
        ft = ft.sigmoid()
        rt = rt.sigmoid()
        if act == 'tanh':
            X = X.tanh()
        elif act == 'relu':
            X = X.relu()
        elif act == 'sigmoid':
            X = X.sigmoid()
        elif act == 'none':
            X = X
        return Wx,ft,rt,X

    def forward(self,x,reverse=False):
        Wx,ft,rt,X = self.SRUconvGates(x)
        Wxs = Wx.split(1, 2)
        fts = ft.split(1, 2)
        rts = rt.split(1, 2)
        #rt2s = rt2.split(1,2)
        #xs = torch.repeat_interleave(x, repeats=self.out_channels, dim=1).split(1,2)
        xs = X.split(1,2)
        hts = []
        #print(reverse)
        if not reverse:
            Ct = None
            ht = None
            for time, (xs_,Wxs_,fts_,rts_) in enumerate(zip(xs,Wxs,fts,rts)):
                Ct,ht = self.SRUconvLayer(xs_,Ct,Wxs_,fts_,rts_)
                hts.append(ht)
        else:
            Ct = None
            ht = None
            for time, (xs_,Wxs_,fts_,rts_) in enumerate(zip(reversed(xs), reversed(Wxs), reversed(fts),reversed(rts))):  # split along timestep
                Ct,ht = self.SRUconvLayer(xs_,Ct,Wxs_,fts_,rts_)
                hts.insert(0, ht)
        hts = torch.cat(hts, dim=2)
        hts = self.conv_before_cc(hts)
        for j in range(2):
            for i in range(hts.shape[2]):
                hts[:,:,i,:,:]= self.cc_model(hts[:,:,i,:,:])
        return hts# +X

class DoSRUconv_convbnrelu_ccnet(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, s=1, p=1, bias=False,bn=True,model='unknown'):
        super(DoSRUconv_convbnrelu_ccnet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if model == 'conv':
            self.conv = BasicConv3d(self.in_channels,self.out_channels*6,k, s, p, bias=bias,bn=bn)
            #self.convX = torch.nn.Conv3d(self.in_channels,self.out_channels,1,1,0)
        elif model == 'deconv':
            self.conv = BasicDeConv3d(self.in_channels,self.out_channels*6,k, s, p, bias=bias,bn=bn)
            #self.convX = torch.nn.ConvTranspose3d(self.in_channels,self.out_channels,1,1,0)
        self.cc_model = CC_module(self.out_channels)
        self.conv_before_cc = Conv3dBNReLU(self.out_channels,self.out_channels,k=(1,3,3), s=1, p=(0,1,1))

    def SRUconvGates(self,input):
        gates = self.conv(input)
        Wx,ft,ft2,rt,rt2,X = gates.split(split_size=self.out_channels,dim=1)
        #X = self.convX(input)
        Wx = Wx.tanh()
        ft = ft.sigmoid()
        ft2 = ft2.sigmoid()
        rt = rt.sigmoid()
        rt2 = rt2.sigmoid()
        if act == 'tanh':
            X = X.tanh()
        elif act == 'relu':
            X = X.relu()
        elif act == 'sigmoid':
            X = X.sigmoid()
        elif act == 'none':
            X = X
        return Wx,ft,ft2,rt,rt2,X

    def SRUconvLayer(self,xs_,Ct,Wxs_,fts_,rts_):
        if Ct is None:
            Ct = 1-fts_
        else:
            Ct = fts_*Ct + (1-fts_)*Wxs_
        #ht = rts_*gs_*Ct+ (1-rts_)* xs_
        ht = rts_*Ct+ (1-rts_)* xs_
        return Ct,ht

    def forward(self,x):
        Wx,ft,ft2,rt,rt2,X = self.SRUconvGates(x)
        Wxs = Wx.split(1, 2)
        fts = ft.split(1, 2)
        ft2s = ft2.split(1,2)
        rts = rt.split(1, 2)
        rt2s = rt2.split(1,2)
        xs = X.split(1,2)
        Ct = None
        ht = None
        htl = []
        
        for time, (xs_,Wxs_,fts_,rts_) in enumerate(zip(xs,Wxs,fts,rts)):
            Ct,ht = self.SRUconvLayer(xs_,Ct,Wxs_,fts_,rts_)
            htl.append(ht)
        htl = torch.cat(htl, dim=2)
        
        Ct = None
        ht = None
        htr = []
        
        for time, (xs_,Wxs_,fts_,rts_) in enumerate(zip(reversed(xs), reversed(Wxs), reversed(ft2s),reversed(rt2s))):  # split along timestep
            Ct,ht = self.SRUconvLayer(xs_,Ct,Wxs_,fts_,rts_)
            htr.insert(0, ht)
        htr = torch.cat(htr, dim=2)
        #end = torch.cat([htl,htr],dim=2)
        out  = htr + htl
        out = self.conv_before_cc(out)
        for j in range(2):
            for i in range(out.shape[2]):
                out[:,:,i,:,:]= self.cc_model(out[:,:,i,:,:])
        return out# +X

class convbnrelu_ccnet_SiSRUconv(nn.Module):
    def __init__(self, in_channels, out_channels, k, s, p,upsample=(1,2,2), bias=False,bn=True,model='unknown'):
        super(convbnrelu_ccnet_SiSRUconv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if model == 'conv':
            self.conv = BasicConv3d(self.in_channels,self.out_channels*4,k, s, p, bias=False,bn=bn)
        elif model == 'deconv':
            self.conv = BasicDeConv3d(self.in_channels,self.out_channels*4,k, s, p, bias=False,bn=bn)
        elif model == 'upsampleconv':
            self.conv = BasicUpsampleConv3d(self.in_channels, self.out_channels*4, k, s, p, upsample=upsample, bn=bn)
        self.cc_model = CC_module(self.in_channels)
        self.convbnrelu = Conv3dBNReLU(self.in_channels,self.in_channels,k=(1,3,3), s=1, p=(0,1,1))
        
    def SRUconvLayer(self,xs_,Ct,Wxs_,fts_,rts_):
        if Ct is None:
            Ct = 1-fts_
        else:
            Ct = fts_*Ct + (1-fts_)*Wxs_
        #ht = rts_*gs_*Ct+ (1-rts_)* xs_
        ht = rts_*Ct+ (1-rts_)* xs_
        return Ct,ht

    def SRUconvGates(self,input):
        gates = self.conv(input)
        Wx,ft,rt,X = gates.split(split_size=self.out_channels,dim=1)
        #X = self.convX(input)
        Wx = Wx.tanh()
        ft = ft.sigmoid()
        rt = rt.sigmoid()
        if act == 'tanh':
            X = X.tanh()
        elif act == 'relu':
            X = X.relu()
        elif act == 'sigmoid':
            X = X.sigmoid()
        elif act == 'none':
            X = X
        return Wx,ft,rt,X

    def forward(self,x,reverse=False):
        x = self.convbnrelu(x)
        for j in range(2):
            for i in range(x.shape[2]):
                x[:,:,i,:,:]= self.cc_model(x[:,:,i,:,:])
        Wx,ft,rt,X = self.SRUconvGates(x)
        Wxs = Wx.split(1, 2)
        fts = ft.split(1, 2)
        rts = rt.split(1, 2)
        #rt2s = rt2.split(1,2)
        #xs = torch.repeat_interleave(x, repeats=self.out_channels, dim=1).split(1,2)
        xs = X.split(1,2)
        hts = []
        #print(reverse)
        if not reverse:
            Ct = None
            ht = None
            for time, (xs_,Wxs_,fts_,rts_) in enumerate(zip(xs,Wxs,fts,rts)):
                Ct,ht = self.SRUconvLayer(xs_,Ct,Wxs_,fts_,rts_)
                hts.append(ht)
        else:
            Ct = None
            ht = None
            for time, (xs_,Wxs_,fts_,rts_) in enumerate(zip(reversed(xs), reversed(Wxs), reversed(fts),reversed(rts))):  # split along timestep
                Ct,ht = self.SRUconvLayer(xs_,Ct,Wxs_,fts_,rts_)
                hts.insert(0, ht)
        hts = torch.cat(hts, dim=2)
        return hts# +X

class NSSNN(nn.Module):
    def __init__(self, in_channels, channels, num_half_layer, sample_idx, has_ad=True, bn=True, plain=False,if_fe=True):
        print(if_fe)
        super(NSSNN, self).__init__()
        assert sample_idx is None or isinstance(sample_idx, list)

        self.enable_ad = has_ad
        if sample_idx is None: sample_idx = []
        self.feature_extractor = DoSRUconv(in_channels, channels,k=3, s=1, p=1,bn=bn, model='conv')

        self.encoder = SRU3DEncoder(channels, num_half_layer, sample_idx, has_ad=has_ad, bn=bn, plain=plain)
        self.decoder = SRU3DDecoder(channels*(2**len(sample_idx)), num_half_layer, sample_idx, has_ad=has_ad, bn=bn, plain=plain)
        
        self.reconstructor = DoSRUconv(channels, in_channels,k=3, s=1, p=1, bias=True, bn=bn, model='deconv')
        self.if_fe = if_fe
        #self.cc_2d = CC_module(1)


    def forward(self, x):
        if self.if_fe:
            xs = [x]
            out = self.feature_extractor(xs[0])
            # print(out.shape)
            
            xs.append(out)
            if self.enable_ad:    
                #print(out.shape)        
                out, reverse = self.encoder(out, xs, reverse=False)
                #print(out.shape)

                out = self.decoder(out, xs, reverse=(reverse))
                #print(out.shape)
            else:
                out = self.encoder(out, xs)
                #print(out.shape)
                out = self.decoder(out, xs)
                #print(out.shape)
            out = out +xs.pop()
            #print(out.shape)
            out = self.reconstructor(out)
            # print(out.shape)
            out = out + xs.pop()
            
            #print(out.shape)
            return out
        else:
            xs = [x]
            out = (xs[0])
            #xs.append(out)
            if self.enable_ad:    
                #print(out.shape)        
                out, reverse = self.encoder(out, xs, reverse=False)
                #print(out.shape)
                out = self.decoder(out, xs, reverse=(reverse))
                #print(out.shape)
            else:
                out = self.encoder(out, xs)
                #print(out.shape)
                out = self.decoder(out, xs)
                #print(out.shape)
            out = out + xs.pop()
            #print(out.shape)
            #out = (out)
            #print(out.shape)
            #out = out + xs.pop()
            #print(out.shape)
            return out
            
class SRU3DEncoder(nn.Module):
    def __init__(self, channels, num_half_layer, sample_idx, has_ad=True, bn=True, plain=False):
        super(SRU3DEncoder, self).__init__()
        # Encoder        
        self.layers = nn.ModuleList()
        self.enable_ad = has_ad
        for i in range(num_half_layer):
            if i not in sample_idx:
                encoder_layer = convbnrelu_ccnet_SiSRUconv(channels, channels,k=3, s=1, p=1, bn=bn,model='conv')
            else:
                if not plain:
                    encoder_layer = SiSRUconv(channels, 2*channels, k=3, s=(1,2,2), p=1, bn=bn,model='conv')
                else:
                    encoder_layer = SiSRUconv(channels, 2*channels, k=3, s=(1,1,1), p=1, bn=bn,model='conv')

                channels *= 2
            self.layers.append(encoder_layer)

    def forward(self, x, xs, reverse=False):
        if not self.enable_ad:            
            num_half_layer = len(self.layers)
            for i in range(num_half_layer-1):
                x = self.layers[i](x)   
                # print(x.shape)         
                xs.append(x)            
            x = self.layers[-1](x)        
            # print(x.shape)
            return x
        else:
            num_half_layer = len(self.layers)
            for i in range(num_half_layer-1):
                x = self.layers[i](x, reverse=reverse)
                # print(x.shape)
                reverse = not reverse
                xs.append(x)            
            x = self.layers[-1](x, reverse=reverse)
            # print(x.shape)
            reverse = not reverse
            
            return x, reverse

class SRU3DDecoder(nn.Module):
    def __init__(self, channels, num_half_layer, sample_idx, is_2d=False, has_ad=True, bn=True, plain=False):
        super(SRU3DDecoder, self).__init__()
        # Decoder
        self.layers = nn.ModuleList()
        self.enable_ad = has_ad
        for i in reversed(range(num_half_layer)):
            if i not in sample_idx:
                    decoder_layer = SiSRUconv(channels, channels,k=3, s=1, p=1, bn=bn,model='deconv')
            else:
                    if not plain:
                        decoder_layer = SiSRUconv(channels, channels//2,k=3, s=1, p=1, upsample=(1,2,2), bn=bn,model = 'upsampleconv')
                    else:
                        decoder_layer = SiSRUconv(channels, channels//2, k=3, s=1, p=1, bn=bn,model='deconv')
                    
                    channels //= 2
            self.layers.append(decoder_layer)


    def forward(self, x, xs, reverse=False):        
        if not self.enable_ad:
            num_half_layer = len(self.layers)
            x = self.layers[0](x)
            # print(x.shape)
            for i in range(1, num_half_layer):
                x = x + xs.pop()
                x = self.layers[i](x)
                # print(x.shape)
            return x
        else:
            num_half_layer = len(self.layers)
            x = self.layers[0](x, reverse=reverse)
            # print(x.shape)
            reverse = not reverse
            for i in range(1, num_half_layer):
                x = x + xs.pop()
                x = self.layers[i](x, reverse=reverse)

                # tmp = x
                # out_np = tmp[0,11,:,:,:].permute(1,2,0).cpu().numpy()
                # io.savemat('/nas_data/fugym/projects_python/SRU3D/results/nssrnn/nssrnnfeature'+str(i)+'.mat',{'nssrnnfeature':out_np})
                # print(x.shape)
                reverse = not reverse
            return x
