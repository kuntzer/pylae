import numpy as np
import pylab as plt
import pylae
import pylae.utils as u
import tools
import os
from pylae.utils_mnist import load_MNIST_images

#=================================================================================================#

network_name = 'output/ae_rmse/test'
path_in_dataset = 'output/psf_nonoise/gaussians.pkl'
scale = 0.0012

m = 5000
ids_train = range(m)
ids_validation = range(m, 7000)
ids_test = range(7500, 10000)




#=================================================================================================#

print 'Datasets...',

dataset = u.readpickle(path_in_dataset)
#nonoise = copy.copy(dataset)

#dataset += np.random.normal(scale=scale, size=dataset.shape)

dataset, _, _ = u.normalise(dataset)
#nonoise, _, _ = u.normalise(nonoise)

train = dataset[ids_train]
#train_nonoise = nonoise[ids_train]

test = dataset[ids_test]
#test_nonoise = nonoise[ids_test]


images = load_MNIST_images('data/mnist/train-images.idx3-ubyte')
images = images.T

train = images[0:5000]#[0:50000]
test = images[50000:]
print 'loaded.'

#=================================================================================================#

def tire(arr, m):
	arr = np.reshape(arr, (1, arr.shape[0], arr.shape[1]))
	arr = np.repeat(arr, m, axis=0)
	return arr

def central_coeffs(size, x, y):
	
	X, Y = np.meshgrid(np.arange(size, dtype=np.float), np.arange(size, dtype=np.float))

	X = tire(X, m)
	Y = tire(Y, m)

	for ii in range(np.shape(x)[0]):
		X[ii] -= x[ii]
		Y[ii] -= y[ii]
		
	nz = X.shape[1]
	
	X = np.reshape(X, (m, nz * nz))
	Y = np.reshape(Y, (m, nz * nz))
		
	return X, Y

n_comp = 50
dA = pylae.rmse_layer.Layer(n_comp, 'SIGMOID', 'SIGMOID', 0, 400, corruption=None)

size = 28

path_in_meas = os.path.join(network_name, 'measurments.pkl')
if not os.path.exists(network_name):
	os.makedirs(network_name)

print 'Measuring dataset...',
do_measure = False
if do_measure:
	dt = np.reshape(train, [m, size, size])
	x, y = np.meshgrid(np.arange(24, dtype=np.float), np.arange(24, dtype=np.float))
	x -= 12.5
	y -= 12.5
	measures = []
	for ii in range(m):
		I = dt[0]

		#sI = np.sum(I)
		e1 = (np.sum(x*x*I) - np.sum(y*y*I)) #/ (np.sum(x*x*I) + np.sum(y*y*I)) / 2.
		e2 = (np.sum(x*y*I)) #/ (np.sum(x*x*I) + np.sum(y*y*I))
		r2 = (np.sum(x*x*I) + np.sum(y*y*I))
		f = np.sum(I)
		measures.append([e1, e2, r2, f])
	measures = np.asarray(measures)
	u.writepickle(measures, path_in_meas)
	print 'done.'
else:
	measures = u.readpickle(path_in_meas)
	print 'loaded.'

E1 = measures[:,0]
E2 = measures[:,1]
R2 = measures[:,2]
F =  measures[:,3]
#x = measures[:,4]
#y = measures[:,5]

c = 12.5*np.ones_like(E1)
X, Y = central_coeffs(size, c, c)



dA.train(train, show=False, regularisation=0.01, sparsity=[3., 0.01])#[0.0,0.3])


tiles=u.tile_raster_images(
		X=dA.weights.T,
		img_shape=(size, size), tile_shape=(10, 10),
		tile_spacing=(1, 1))

tiles=u.tile_raster_images(
		X=dA.weights.T,
		img_shape=(size, size), tile_shape=(10, 10),
		tile_spacing=(1, 1))

plt.figure()
plt.imshow(tiles, interpolation='None')
plt.title('cost=%1.1f' % (u.cross_entropy(train, dA.full_feedforward(train))))
plt.show()


dA.plot_train_history()

reconstruc = dA.full_feedforward(test)

pca = u.compute_pca(train, n_components=n_comp)
recont_pca = pca.transform(test)
recont_pca = pca.inverse_transform(recont_pca)
recont_pca, _,_ = u.normalise(recont_pca) # PCA IS NOT NORMALISED!

# To avoid numerical issues...
recont_pca *= 0.999999
recont_pca += 1e-10

print 'AE cost: ', u.cross_entropy(test, reconstruc)
print 'PCA cost: ', u.cross_entropy(test, recont_pca)

varrent = 0
varrentpca = 0

for ii in range(np.shape(test)[0]):
	true = test[ii]
	approx = reconstruc[ii]
	approxpca = recont_pca[ii]
	
	div = np.linalg.norm(true)
	div = div * div
	
	norm = np.linalg.norm(true - approx)
	norm = norm * norm

	varrent += norm / div / np.shape(test)[0]
	
	norm = np.linalg.norm(true - approxpca)
	norm = norm * norm
	varrentpca += norm / div / np.shape(test)[0]

print "Variance retention for ae:", varrent
print "Variance retention for pca:", varrentpca

size = int(np.sqrt(np.shape(reconstruc)[1]))
# Show the original image, the reconstruction and the residues for the first 10 cases
for ii in range(25):
	img = test[ii].reshape(size,size)
	recon = reconstruc[ii].reshape(size,size)
	recont, _ ,_ = u.normalise(recont_pca[ii].reshape(size,size))
	recon_cd1 = reconstruc[ii].reshape(size,size)

	plt.figure(figsize=(9,6))
	
	plt.subplot(2, 3, 1)
	plt.imshow((img), interpolation="nearest")
	
	plt.subplot(2, 3, 2)
	plt.title("Gradient Desc.")
	plt.imshow((recon), interpolation="nearest")
	plt.subplot(2, 3, 3)
	plt.imshow((img - recon), interpolation="nearest")
	plt.title("Gradient Desc. residues")
	
	plt.subplot(2, 3, 1)
	#plt.imshow(dA._corrupt(img), interpolation="nearest")
		
	plt.subplot(2, 3, 5)
	plt.imshow((recont), interpolation="nearest")
	plt.title("PCA")
	plt.subplot(2, 3, 6)
	iim, _, _ = u.normalise(img)
	plt.imshow(iim - recont, interpolation="nearest")
	plt.title("PCA residues")

plt.show()