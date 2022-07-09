from array import array
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    x = np.load(filename)
    x = x - np.mean(x, axis=0)
    return x

def get_covariance(dataset):
    t = np.dot(np.transpose(dataset), dataset)
    t /= (len(dataset)-1)
    return t

def get_eig(S, m):
    evalue, evector= eigh(S, subset_by_index=[len(S)-m,len(S)-1])
    Lambda = np.diag(np.flipud(evalue))
    U = np.fliplr(evector)
    return Lambda, U
  
def get_eig_prop(S, prop):
    evalue, evector= eigh(S)
    total = sum (evalue)
    evalue, evector= eigh(S, subset_by_value=[total*prop,total])
    Lambda = np.diag(np.flipud(evalue))
    U = np.fliplr(evector)
    return Lambda, U
   
def project_image(image, U):
    t= np.transpose(U)
    result = np.zeros(len(U))
    for i in range(len(U[0])):
        result += np.multiply(np.dot(t[i],image),t[i])
    return result
   
def display_image(orig, proj):
    
    before =  np.transpose(orig.reshape(32,32))
    after = np.transpose(proj.reshape(32,32))
    fig, ax = plt.subplots(1, 2)
    
    image1= ax[0].imshow(before, aspect='equal')
    ax[0].set_title("Original")
    ax[0].set_xticks(range(0,31,10))
    ax[0].set_yticks(range(0,31,5))
    fig.colorbar(image1, ax= ax[0], ticks=range(-25, 126,25), shrink = 0.5)
    
    image2 = ax[1].imshow(after, aspect='equal')
    ax[1].set_title("Projection")
    ax[1].set_xticks(range(0,31,10))
    ax[1].set_yticks(range(0,31,5))
    fig.colorbar(image2, ax = ax[1], ticks=range(20,81,20),shrink = 0.5)
    
    plt.show()

 
x = load_and_center_dataset('YaleB_32x32.npy')
S = get_covariance(x)
Lambda, U = get_eig(S, 2)
projection = project_image(x[0], U)
display_image(x[0], projection)
    
