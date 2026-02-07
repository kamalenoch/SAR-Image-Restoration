import numpy as np

def ca_cfar(img, guard=2, train=8, pfa=1e-5):
    h, w = img.shape
    out = np.zeros_like(img)
    k = 2*(guard+train)+1
    n = k*k - (2*guard+1)**2
    alpha = n*(pfa**(-1/n)-1)
    pad = np.pad(img, guard+train)

    for i in range(h):
        for j in range(w):
            win = pad[i:i+k, j:j+k]
            guard_area = pad[i+train:i+train+2*guard+1, j+train:j+train+2*guard+1]
            noise = (win.sum()-guard_area.sum())/n
            if img[i,j] > alpha*noise:
                out[i,j]=1
    return out
