import glob
import scipy.io as sio
import numpy as np
# import cv2
# import yaml
import h5py

if __name__ == '__main__':

    for mat in glob.glob('bsifcpp/texturefilters/ICAtextureFilters*.mat'):
        # read one MATLAB file
        data = sio.loadmat(mat)
        npmat = np.array(data['ICAtextureFilters'])

        yamlname = mat.replace('.mat','.hdf5')
        print("Saving", yamlname)
        # with open(yamlname,'w') as f:
        #     f.write("%YAML:1.0\n")
        #     yaml.dump({"ICAtextureFilters": npmat}, f)

        # try:
        #     f = cv2.FileStorage(yamlname, cv2.FILE_STORAGE_READ)
        #     f['ICAtextureFilters'] = npmat
        # finally:
        #     f.release()

        with h5py.File(yamlname, 'w') as f:
            dset = f.create_dataset("ICAtextureFilters", npmat.shape, dtype='f8')
            dset[:] = npmat

        # break
    print("Done.")