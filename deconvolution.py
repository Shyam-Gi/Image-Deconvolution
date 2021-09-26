import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import filters


def dft_matrix(N):
    f = np.zeros((N, N), dtype='complex')
    j = (-1)**0.5
    w = np.exp(-j*2*np.pi/N)
    for a in range(0, N):
        for b in range(0, N):
            f[a, b] = w**(a*b)
    return f


def idft_matrix(N):
    f = np.zeros((N, N), dtype='complex')
    j = (-1)**0.5
    w = np.exp(j*2*np.pi/N)
    for a in range(0, N):
        for b in range(0, N):
            f[a, b] = w**(a*b)
    return f


def dft(X):
    N, M = X.shape
    f1 = np.zeros((N, N), dtype='complex')
    f2 = np.zeros((M, M), dtype='complex')
    f1 = dft_matrix(N)
    f2 = dft_matrix(M)
    X_dft = np.dot(f1, np.dot(X, f2))
    return X_dft


def idft(X):
    N, M = X.shape
    f_1 = np.zeros((N, N), dtype='complex')
    f_2 = np.zeros((M, M), dtype='complex')
    f_1 = idft_matrix(N)
    f_2 = idft_matrix(M)
    V = 1/(N*M) * np.dot(f_1, np.dot(X, f_2))
    return V


def conv_2D(x, h):
    return np.fft.ifftn(np.fft.fftn(x)*np.fft.fftn(h))


def dot(a, b):
    return np.sum(a*b)


def cg(h, m, lam):
    def get_derv(Sz):
        Dx = np.zeros(Sz)
        Dx[0, 0] = -1.
        Dx[0, -1] = 1.
        Dy = Dx.T
        FDx = np.fft.fftn(Dx)
        FDy = np.fft.fftn(Dy)
        return (FDx*np.conj(FDx)+FDy*np.conj(FDy))

    def apply_derv(x):

        Fq = get_derv(x.shape)
        return np.fft.ifftn(Fq*np.fft.fftn(x))

    def A(x):
        # change here to make proper a00
        return conv_2D(conv_2D(h,h_) ,x)+ lam*apply_derv(x)
    x0 = m
    b_ = conv_2D(b,h_)
   # change here to make b=h-*x 00
    r0 = b_-A(x0)
    r = r0
    p = r
    x = x0
    Niter = 100
    for i in range(Niter):
        alpha = dot(r, r)/dot(p, A(p))
        x = x+alpha*p
        r_p = r
        r = r-alpha*A(p)
        beta = dot(r, r)/dot(r_p, r_p)
        p = r+beta*p
        u = np.linalg.norm(r)
        if u < 10**-3:
            break
        print(u)
    return x

astro_image = skimage.data.hubble_deep_field()
a, c, d = astro_image.shape
e = a if a < c else c
astro_image = astro_image[0:e, 0:e, 0:d]
a = np.zeros((e, e))
a[e//2, e//2] = 1
# filter
h = filters.gaussian(a, sigma=(3, 3), multichannel='false')
h_ = np.fft.ifftn(np.conjugate(np.fft.fftn(h)))
xs = np.zeros((e, e, d))
for i in range(0, 3):
    astro_ = astro_image[:, :, i]  # selects 0 channel
    astro_ = astro_/astro_.max()
    b = filters.gaussian(astro_, sigma=(3, 3), multichannel='false')
    # blurred image
    x = np.zeros((e, e))
    # applying CGradient
    x_ = cg(h, x, 0.08)
    xx = np.asarray(np.real(x_), dtype='float')
    xx = xx/xx.max()
    xs[:, :, i] = xx
aa=np.zeros((e,e,d))
aa[0:e//2,0:e//2,0:d]=xs[e//2:e,e//2:e,0:d]
aa[0:e//2,e//2:e,0:d]=xs[e//2:e,0:e//2,0:d]
aa[e//2:e,e//2:e,0:d]=xs[0:e//2,0:e//2,0:d]
aa[e//2:e,0:e//2,0:d]=xs[0:e//2,e//2:e,0:d]
plt.imshow(aa)
plt.show()
