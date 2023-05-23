import numpy as np
import faiss
from matplotlib import pyplot
# generate data

# different variance in each dimension
x0 = faiss.randn((1000, 16)) * (1.2 ** -np.arange(16))

# random rotation
R, _ = np.linalg.qr(faiss.randn((16, 16)))   
x = np.dot(x0, R).astype('float32')
# compute and visualize the covariance matrix
xc = x - x.mean(0)
cov = np.dot(xc.T, xc) / xc.shape[0]
_ = pyplot.imshow(cov)
pyplot.show()

# map the vectors back to a space where they follow a unit Gaussian
L = np.linalg.cholesky(cov)
mahalanobis_transform = np.linalg.inv(L)
y = np.dot(x, mahalanobis_transform.T)

# covariance should be diagonal in that space...
yc = y - y.mean(0)
ycov = np.dot(yc.T, yc) / yc.shape[0]
_ = pyplot.imshow(ycov)
pyplot.show()


# perform L2 search in the tranformed space 
index = faiss.IndexFlatL2(16)
index.add(y[:500])
D, I = index.search(y[500:], 10)