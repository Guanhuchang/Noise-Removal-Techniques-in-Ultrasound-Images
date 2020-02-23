import cv2
from itertools import product
from PIL import Image
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA

class Robust_pca:

    #initialize S0=Y0=L0=0
    #initialize lambda, mu , mu(inverse) , tolerance,
    def __init__(self,M):
        self.M=M
        self.S=np.zeros(self.M.shape)
        self.Y=np.zeros(self.M.shape)
        self.L=np.zeros(self.M.shape)
        self.lam=1/np.sqrt(np.max(self.M.shape))
        self.mu=10*self.lam
        self.mu_inv=1/(self.mu)
        self.tolerance=1e-8
        self.max_iter=800


    def S_function(self,M,tau):
        result=np.sign(M)*np.maximum(np.abs(M)-tau,0)
        return result


    def D_function(self,M,tau):
        U,S,V=np.linalg.svd(M,full_matrices=False)
        result_s=self.S_function(S,tau)
        US=np.dot(U,np.diag(result_s))
        result=np.dot(US,V)
        return result


    def generate_pca(self):

        Sk=self.S
        Yk=self.Y
        Lk=self.L
        err=np.Inf

        #run loop until reach max iteration or converged
        for i in range (0,self.max_iter):

            Lk=self.D_function(self.M-Sk+self.mu_inv*Yk,self.mu_inv)

            Sk=self.S_function(self.M-Lk+self.mu_inv*Yk,self.mu_inv*self.lam)

            Yk=Yk+self.mu*(self.M-Lk-Sk)
            #compute the error using Frobenius norm
            err=np.linalg.norm(self.M-Lk-Sk,'fro')/np.linalg.norm(self.M,'fro')

            #check convergence
            if err<self.tolerance:
                break

        self.L=Lk
        self.S=Sk
        return Lk,Sk

def main():


    i=1
    for i in range (1,10):

        #Load input data as gray image
        X = np.array(Image.open(str(i)+".jpeg").convert('L'))

        #implement RPCA
        rpca_implement=Robust_pca(X)
        L,S=rpca_implement.generate_pca()
        cv2.imwrite(str(i)+'_RPCA.png',L)

main()
