import cv2
from itertools import product
from PIL import Image
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA


def main():
    #Load input data as gray image
    i=1
    for i in range (1,10):
        X = np.array(Image.open(str(i)+".jpeg").convert('L'))

        #implement PCA
        sklearn_pca=sklearnPCA()
        Xproj=sklearn_pca.fit_transform(X)
        Xdenoised=sklearn_pca.inverse_transform(Xproj)
        pca_implement=Xdenoised.astype(np.uint8)

        cv2.imwrite(str(i)+'_PCA.png',pca_implement)

main()
