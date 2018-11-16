import numpy as np
import bsif
import cv2
import csv
import sys

if len(sys.argv)==2:

    # imin = np.array([[10, 20],[30,40],[50,60]], dtype=np.uint8)
    dims=np.array([7,7,12], dtype=np.uint8)

    imin = cv2.imread(sys.argv[1], 0)
    # imin = imin[:,:,2].astype(np.uint8)
    imout = np.zeros_like(imin)
    # print("python array:", imin[0:10,0:10])

    #np.savetxt("imagepython.csv", imin, delimiter=',')

    # cv2.imshow("python",imin)

    imout = bsif.extract(imin, imout, dims)

    # result will come as a float, so we'll scale it between 0 to 1, according to
    # the depth of the filter that was used
    imout = imout / (2**dims[2])

    # cv2.imshow("pythonout",imout)
    # cv2.waitKey(0)
    cv2.imwrite('./output/python.png',imout*255)
