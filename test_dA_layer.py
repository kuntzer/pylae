import numpy as np
import pylab as plt
import pylae
import pylae.utils as utils
from pylae.utils_mnist import load_MNIST_images


images = load_MNIST_images('data/mnist/train-images.idx3-ubyte')
images = images.T

patches = images[0:1000]#[0:50000]
test = images[45000:50000]

corruption = None#0.051

n_comp = 100
dA = pylae.dA_layer.Layer(n_comp, 'SIGMOID', 'SIGMOID', 0, 200, corruption=corruption)#[0.0,0.3])
print 'STARTING TRAINING'
dA.train(patches)

tiles=utils.tile_raster_images(
		X=dA.weights.T,
		img_shape=(28, 28), tile_shape=(10, 10),
		tile_spacing=(1, 1))

tiles=utils.tile_raster_images(
		X=dA.weights.T,
		img_shape=(28, 28), tile_shape=(10, 10),
		tile_spacing=(1, 1))

plt.figure()
plt.imshow(tiles, interpolation='None')
plt.title('corr=%s, cost=%1.1f' % (dA.corruption, utils.cross_entropy(patches, dA.full_feedforward(patches))))
plt.show()


dA.plot_train_history()

size = 28
reconstruc = dA.full_feedforward(test)

pca = utils.compute_pca(patches, n_components=n_comp)
recont_pca = pca.transform(test)
recont_pca = pca.inverse_transform(recont_pca)
recont_pca, _,_ = utils.normalise(recont_pca) # PCA IS NOT NORMALISED!

# To avoid numerical issues...
recont_pca *= 0.999999
recont_pca += 1e-10

print 'AE cost: ', utils.cross_entropy(test, reconstruc)
print 'PCA cost: ', utils.cross_entropy(test, recont_pca)

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
	recont, _ ,_ = utils.normalise(recont_pca[ii].reshape(size,size))
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
	iim, _, _ = utils.normalise(img)
	plt.imshow(iim - recont, interpolation="nearest")
	plt.title("PCA residues")

plt.show()