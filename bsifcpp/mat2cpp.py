import glob
import scipy.io as sio
import numpy as np

if __name__ == '__main__':

    with open('bsifcpp/filters.h','w') as f:
        f.write('#ifndef __ICAFILTERS__\n')
        f.write('#define __ICAFILTERS__\n\n')

        f.write('// file with hard-coded ICAFilters, auto-generated \n')
        f.write('// ****** PLEASE DO NOT EDIT!! *****\n')

        for mat in glob.glob('bsifcpp/texturefilters/ICAtextureFilters_*.mat'):
            # read one MATLAB file
            data = sio.loadmat(mat)
            npmat = np.array(data['ICAtextureFilters'])

            isize, jsize, ksize = npmat.shape

            # take the file name as the variable name
            varname = 'filter_{}_{}_{}'.format(isize, jsize, ksize)
            f.write('double ' + varname + '[] = {\n')

            # iterates through the last dimension
            for i in range(isize):
                for j in range(jsize):
                    for k in range(ksize):
                        f.write(str(npmat[i,j,k]) + ',')
            f.write('};\n')
            f.write('\n')

        f.write('#endif\n')

    #     break
    # print("Done.")