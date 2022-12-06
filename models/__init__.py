from .qrnn import REDC3D
from .qrnn import QRNNREDC3D
#attention
from .sru.sru3d import SRUREDC3D
from models.nssnn.nssnn import NSSNN

"""Define commonly used architecture"""

def nssnn():
    net = NSSNN(1,16,5,[1,3],has_ad=True,bn=False)
    net.use_2dconv = False
    net.bandwise = False
    return net

def residualunet3d():
    net = REDC3D(1, 16, 5, 3)
    net.use_2dconv = False
    net.bandwise = False
    return net

def nssnn_7layers():
    net = NSSNN(1, 16, 7, [1,3,5], has_ad=True,bn=False)
    net.use_2dconv = False
    net.bandwise = False
    return net

def nssnn_3layers():
    net = NSSNN(1, 16, 3, [1], has_ad=True,bn=False)
    net.use_2dconv = False
    net.bandwise = False
    return net

def sru3d_nobn():
    net = SRUREDC3D(1, 16, 5, [1,3], has_ad=True,bn=False)
    net.use_2dconv = False
    net.bandwise = False
    return net

def qrnn3d():
    net = QRNNREDC3D(1, 16, 5, [1,3], has_ad=True,bn=True)
    net.use_2dconv = False
    net.bandwise = False
    return net

def qrnn3d_nobn():
    net = QRNNREDC3D(1, 16, 5, [1,3], has_ad=True,bn=False)
    net.use_2dconv = False
    net.bandwise = False
    return net

def qrnn2d():
    net = QRNNREDC3D(1, 16, 5, [1,3], has_ad=True, is_2d=True)
    net.use_2dconv = False
    net.bandwise = False
    return net
    
