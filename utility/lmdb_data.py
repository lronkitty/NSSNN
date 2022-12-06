"""Create lmdb dataset"""
from termios import XCASE
from util import *
import lmdb
import caffe
import scipy.io

def create_lmdb_train(
    datadir, fns, name, matkey,
    crop_sizes, scales, ksizes, strides,
    load=h5py.File, augment=True,
    seed=2017,trans=1,norm=0,map_size = None):
    """
    Create Augmented Dataset
    """
    def preprocess(data):
        new_data = []
        if trans == 1:
            data[data < 0] = 0
        data = minmax_normalize(data)
        data = np.rot90(data, k=2, axes=(1,2)) # ICVL
        # data = minmax_normalize(data.transpose((2,0,1))) # for Remote Sensing
        # Visualize3D(data)
        if crop_sizes is not None:
            data = crop_center(data, crop_sizes[0], crop_sizes[1])        
        
        for i in range(len(scales)):
            if scales[i] != 1:
                temp = zoom(data, zoom=(1, scales[i], scales[i]))
            else:
                temp = data
            # print(temp.shape)
            temp = Data2Volume(temp, ksizes=ksizes, strides=list(strides[i]))            
            new_data.append(temp)
        new_data = np.concatenate(new_data, axis=0)
        if augment:
             for i in range(new_data.shape[0]):
                 new_data[i,...] = data_augmentation(new_data[i, ...])
                
        return new_data.astype(np.float32)

    np.random.seed(seed)
    scales = list(scales)
    ksizes = list(ksizes)        
    assert len(scales) == len(strides)
    # calculate the shape of dataset
    data = load(datadir + fns[0])
    data= data[matkey]
    if trans:
        data = np.transpose(data,(2,0,1))
    # print(data.shape)
    if map_size is None:
        data = preprocess(data)
        N = data.shape[0]
        
        # print(data.shape)
        map_size = data.nbytes * len(fns) * 1.2
    print('map size (GB):', map_size / 1024 / 1024 / 1024)
    
    #import ipdb#; ipdb.set_trace()
    if os.path.exists(name+'.db'):
        raise Exception('database already exist!')
    env = lmdb.open(name+'.db', map_size=map_size, writemap=True)
    with env.begin(write=True) as txn:
        # txn is a Transaction object
        k = 0
        for i, fn in enumerate(fns):
            try:
                X = load(datadir + fn)[matkey]
                if trans:
                    # print(X.shape)
                    X = np.transpose(X,(2,0,1))
            except:
                print('loading', datadir+fn, 'fail')
                continue    
            X = preprocess(X)
            N = X.shape[0]
            for j in range(N):
                # print(X[j].max(), X[j].min())
                if X[j].min() < -100:
                    continue
                elif X[j].max() == 0:
                    continue
                datum = caffe.proto.caffe_pb2.Datum()
                datum.channels = X.shape[1]
                datum.height = X.shape[2]
                datum.width = X.shape[3]
                # print(X[j].max(), X[j].min())
                if norm == 0:
                    datum.data = X[j].tobytes()
                else:
                    datum.data = minmax_normalize(X[j]).tobytes()
                str_id = '{:08}'.format(k)
                k += 1
                txn.put(str_id.encode('ascii'), datum.SerializeToString())
            print('load mat (%d/%d): %s' %(i,len(fns),fn))
            print(k)

        print('done')


# Create 2mats ICVL training dataset
def create_icvl64_31_2mats():
    print('create icvl64_31...')
    datadir = 'datasets/training/ICVL/train2_2mats/' # your own data address
    fns = os.listdir(datadir) 
    fns = [fn.split('.')[0]+'.mat' for fn in fns]
    
    create_lmdb_train(
        datadir, fns, 'datasets/ICVL64_31_2mats', 'rad',  # your own dataset address
        crop_sizes=(1024, 1024),
        scales=(1, 0.5, 0.25),        
        ksizes=(31, 64, 64),
        strides=[(31, 64, 64), (31, 32, 32), (31, 32, 32)],        
        load=h5py.File, augment=True,trans=0
    )

# Create ICVL training dataset
def create_icvl64_31():
    print('create icvl64_31...')
    datadir = 'datasets/training/ICVL/train/' # your own data address
    fns = os.listdir(datadir) 
    fns = [fn.split('.')[0]+'.mat' for fn in fns]
    
    create_lmdb_train(
        datadir, fns, 'datasets/ICVL64_31', 'rad',  # your own dataset address
        crop_sizes=(1024, 1024),
        scales=(1, 0.5, 0.25),        
        ksizes=(31, 64, 64),
        strides=[(31, 64, 64), (31, 32, 32), (31, 32, 32)],        
        load=h5py.File, augment=True,trans=0
    )

def create_houston64_stride_24_46_norm():
    print('create houston64_46_norm...')
    datadir = 'datasets/training/Houston/' # your own data address
    fns = os.listdir(datadir) 
    fns = [fn.split('.')[0]+'.mat' for fn in fns]
    
    create_lmdb_train(
        datadir, fns, 'datasets/houston64_stride_24_46_norm', 'houston',  # your own dataset address
        crop_sizes=None,
        scales=(1, 0.5, 0.25),        
        ksizes=(46, 64, 64),
        strides=[(10, 24, 24), (5, 12, 12), (5, 12, 12)],   
        load=scipy.io.loadmat, augment=True,norm=1,map_size =15827848396
    )

if __name__ == '__main__':
    # create_icvl64_31_2mats()
    # create_icvl64_31()
    create_houston64_stride_24_46_norm()
    pass
