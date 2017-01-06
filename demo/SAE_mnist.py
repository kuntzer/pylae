import pylab as plt
from sklearn.decomposition import PCA
import sklearn.metrics as metrics
import numpy as np
import os

import pylae
import utils_mnist

import logging

logging.basicConfig(format='PID %(process)06d | %(asctime)s | %(levelname)s: %(name)s(%(funcName)s): %(message)s',level=logging.INFO)

# Loading the data and pre-processing
N_train = 5000
N_test = 1500

images = utils_mnist.load_MNIST_images('mnist-data/train-images-idx3-ubyte')
images = images.T
ids_train = range(N_train)
ids_test = range(N_train, N_train+N_test)

images_train = images[ids_train]
images_test = images[ids_test]

# Preparing the SAE
dA = pylae.dA.AutoEncoder("sae_mnist")

architecture = [128, 64, 8]
layers_activation = ["SIGMOID", "SIGMOID", "SIGMOID"]
cost_fct = 'cross-entropy'

# Define what training we should do
do_pre_train = False
do_train = True
iters = 20

# Layer pre-training
if do_pre_train:
	dA.pre_train(images_train, architecture, layers_activation, iterations=iters, mini_batch=500, cost_fct=cost_fct)
	dA.save()
else:
	dA = pylae.utils.readpickle(os.path.join(dA.filepath, 'ae.pkl'))
	logging.info('pre-trained layers loaded!')

# Fine-tuning
if do_train:
	dA.fine_tune(images_train, iterations=iters, mini_batch=500, cost_fct=cost_fct)
	dA.save()
else:
	dA = pylae.utils.readpickle(os.path.join(dA.filepath, 'ae.pkl'))
	logging.info('Auto-encoders loaded!')
	
dA.verbose = True

pylae.plots.display_train_history(dA)
pylae.plots.display_network(dA, 0)
try:
	pylae.plots.display_network(dA, 1)
except:
	pass
try:
	pylae.plots.display_network(dA, 2)
except:
	pass

# Let's encode and decode the test image to see the result:
images_sae = dA.decode(dA.encode(images_test))

# Compute PCA model for the same training data
pca = PCA(n_components=architecture[-1], whiten=True)
pca.fit(images_train)
pca_enc = pca.transform(images_test)
images_pca = pca.inverse_transform(pca_enc)
images_pca, _, _ = pylae.processing.normalise(images_pca) # PCA IS NOT NORMALISED!

# Print a few metrics
logging.info('** cross-entropy **')
logging.info('AE cost: {:1.3f}'.format(pylae.utils.cross_entropy(images_test, images_sae)))
logging.info('PCA cost: {:1.3f}'.format(pylae.utils.cross_entropy(images_test, images_pca)))

logging.info('** rmse **')
logging.info('AE cost: {:1.2e}'.format(metrics.mean_squared_error(images_test, images_sae)))
logging.info('PCA cost: {:1.2e}'.format(metrics.mean_squared_error(images_test, images_pca)))

logging.info('** explained variance **')
logging.info('AE cost: {:1.2f}'.format(metrics.explained_variance_score(images_test, images_sae)))
logging.info('PCA cost: {:1.2f}'.format(metrics.explained_variance_score(images_test, images_pca)))

# Now show the reconstructed images
size = int(np.sqrt(np.shape(images_test)[1]))
# Show the original image, the reconstruction and the residues for the first 10 cases
for ii in np.random.choice(range(N_test), size=10, replace=False):
	img = images_test[ii].reshape(size,size)
	recon = images_sae[ii].reshape(size,size)
	recont = images_pca[ii].reshape(size,size)

	plt.figure(figsize=(9,6))
	
	plt.subplot(2, 3, 1)
	plt.imshow((img), interpolation="nearest")
	
	plt.subplot(2, 3, 2)
	plt.title("Gradient Desc.")
	plt.imshow((recon), interpolation="nearest")
	plt.subplot(2, 3, 3)
	plt.imshow((img - recon), interpolation="nearest")
	plt.title("Gradient Desc. residues")
		
	plt.subplot(2, 3, 5)
	plt.imshow((recont), interpolation="nearest")
	plt.title("PCA")
	plt.subplot(2, 3, 6)
	plt.imshow((img - recont), interpolation="nearest")
	plt.title("PCA residues")

plt.show()
