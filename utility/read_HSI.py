import torch
import numpy as np

def read_HSI(data,kernel_size=(56,56,31), stride=(15,15,15),device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    import torch
    from models import im2col
    data_shape = []
    data_shape.append(data.shape)
    #device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    # print(data.shape[0],kernel_size[0],stride[0])
    # if data.shape[0] < 31:
    if False:
        pad_x = 31 -data.shape[0]
    else:
        pad_x = (-data.shape[0]+kernel_size[0])%stride[0]
    # pad_x = 15
    pad_y = (-data.shape[1]+kernel_size[1])%stride[1]
    #pad_z =0
    pad_z = (-data.shape[2]+kernel_size[2])%stride[2]
    # print(pad_x,pad_y,pad_z)
    data =np.pad(data,((0, pad_x),(0,  pad_y),(0,pad_z)),'edge')
    # print(data.shape)
    data = torch.from_numpy(data).to(device)
    data_shape.append(data.shape)
    #pad = torch.nn.ReplicationPad3d([0, pad_right, 0,  pad_down,0,pad_z])
    #data = pad(data)
    col_data = im2col.Cube2Col(data.reshape((1,data.shape[0],data.shape[1],data.shape[2])),kernel_size, stride,padding=0,tensorized=True,device=device)
    # print(col_data.shape)
    data_shape.append(col_data.shape)
    #print(col_data[0][:][-1][-1][-1])
    
    col_data = col_data.view(1,kernel_size[0],kernel_size[1],kernel_size[2],col_data.shape[2],col_data.shape[3],col_data.shape[4]) #[1,56,56,31,18,18,1]
    data_shape.append(col_data.shape)
    col_data = col_data.permute(0, 4, 5, 6, 1, 2, 3) #
    data_shape.append(col_data.shape)
    col_data = col_data.view(col_data.shape[0]*col_data.shape[1]*col_data.shape[2]*col_data.shape[3],1,col_data.shape[4],col_data.shape[5],col_data.shape[6])
    data_shape.append(col_data.shape)
    #print(data_shape)
    #test = col_data[:,:,:,:,0,0,0]
    return col_data,data_shape

if __name__ == "__main__":
    import os
    import sys
    import scipy.io as scio
    os.chdir(sys.path[0])
    dataFiles = []
    dataFiles.append('train/4cam_0411-1640-1.mat')
    for dataFile in dataFiles:
        data = scio.loadmat(dataFile)['DataCube']
        test = np.sum(data)
        col_data = read_HSI(data)[0]
        test2 = torch.sum(col_data)
        col_data = col_data.cpu().numpy()
     #data.reshape((1,data.shape[0],data.shape[1],data.shape[2]))
        scio.savemat('col_Data.mat', {'colData':col_data})
    pass