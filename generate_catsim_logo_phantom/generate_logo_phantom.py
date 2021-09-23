import scipy.io as scio
import numpy as np
import json

#------------ phantom definition file (json)
d = {"n_materials":2,  # phantom made of 2 materials, so there're 2 density volume files
"mat_name":["water", "bone"],  # material names
"volumefractionmap_filename":["CatSim_logo.density_1", "CatSim_logo.density_2"],  # filenames of density volumes
"volumefractionmap_datatype":["float", "float"],  # binary data type
"cols":[512, 512],  # dimensions
"rows":[512, 512],
"slices":[1, 1],
"x_size":[0.429688, 0.429688],  # voxel size
"y_size":[0.429688, 0.429688],
"z_size":[42.9688, 42.9688],
"x_offset":[256.5, 256.5],  # center position, in voxels
"y_offset":[256.5, 256.5],
"z_offset":[1, 1]
}

with open('CatSim_logo.json', 'w') as f:
    json.dump(d, f, indent=4)

#------------ density volumes (binary files)
dataFile = 'CatSim_logo_density_volumes.mat'
data = scio.loadmat(dataFile)
print(data['density_volumes'][0, 1].shape)

for ii in range(2):
    binFile = "CatSim_logo.density_%d" %(ii+1)
    des = data['density_volumes'][0, ii].T.copy(order='C')
    with open(binFile, 'wb') as fout:
        fout.write(des)
