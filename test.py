import numpy as np
import faiss
from matplotlib import pyplot
# generate data

# different variance in each dimension
x0 = faiss.randn((1000, 16)) * (1.2 ** -np.arange(16))

# x0 = faiss.read_index('index_file_path2')
# x0 =x0.reconstruct_n(0,x0.ntotal)
# random rotation
R, _ = np.linalg.qr(faiss.randn((16, 16)))   
x = np.dot(x0, R).astype('float32')

x = faiss.read_index('index_file_path2')
x =x.reconstruct_n(0,x.ntotal)

# compute and visualize the covariance matrix
xc = x - x.mean(0)
cov = np.dot(xc.T, xc) / xc.shape[0]

pyplot.imshow(cov[:30,:30])
pyplot.show()


# map the vectors back to a space where they follow a unit Gaussian
L = np.linalg.cholesky(cov)
mahalanobis_transform = np.linalg.inv(L)
y = np.dot(x, mahalanobis_transform.T)

# covariance should be diagonal in that space...
yc = y - y.mean(0)
ycov = np.dot(yc.T, yc) / yc.shape[0]
_ = pyplot.imshow(ycov[:30,:30])
print(ycov)
pyplot.show()

# perform L2 search in the tranformed space 
index = faiss.IndexFlatL2(768)
index.add(y)
t = index.reconstruct(987)
t=t.reshape(1,-1)
D,I = index.search(t,10)
print(D,)
print(I)