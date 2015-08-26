#encoding=UTF-8
import os, struct
import numpy as np

def read_matbin(path):
    if not os.path.exists(path):
        return None

    with open(path, 'rb') as fin:
        (mtype, dims) = struct.unpack('ii', fin.read(8))
        depth = mtype & 7
        elem_size = 1 << (depth / 2)
        sizes = struct.unpack('i' * dims, fin.read(4 * dims))
        mat_size = reduce(lambda x,y:x*y, sizes, 1)
        channels = ((mtype & (511 << 3)) >> 3) + 1
        type_table = ['uint8', 'int8', 'uint16', 'int16', 'int32', 'Float32', 'Float64']
        np_data = np.fromfile(fin, dtype=type_table[depth])
        
        if channels == 1:
            np_data = np.reshape(np_data, sizes)
        else:
            np_data = np.reshape(np_data, sizes + [channels])
    return np_data    

if __name__ == '__main__':
    data = read_matbin('00055.matbin')
    print data[-1,-1,:]
