import numpy as np
import pylab as plt
import pylae
import pylae.utils as utils
from pylae.utils_mnist import load_MNIST_images

network_name = 'ut'

images = load_MNIST_images('data/mnist/train-images.idx3-ubyte')
images = images.T
patches = images[0:8000]
test = images[8000:]

gd = pylae.autoencoder.AutoEncoder(network_name, rbm_type="gd")
architecture = [196]
layers_type = ["SIGMOID", "SIGMOID"]

pre_train = False
train = False
if pre_train:
	gd.pre_train(patches, architecture, layers_type, learn_rate={'SIGMOID':0.05, 'LINEAR':0.05/10.}, 
				initialmomentum=0.53,finalmomentum=0.93, iterations=1, mini_batch=100, regularisation=0.02)
	utils.writepickle(gd.rbms, "%s/gd/rbms.pkl" % network_name)
else:
	rbms = utils.readpickle("%s/gd/rbms.pkl" % network_name)
	gd.set_autoencoder(rbms)
	gd.is_pretrained = True

if train:
	gd.fine_tune(patches, iterations=50, regularisation=0.002, sparsity=0.1, beta=3.)

	gd.save()
else:
	gd = utils.readpickle("%s/gd/ae.pkl" % network_name)

gd.display_network(0)
gd.display_network(1)
plt.show()

enc = gd.encode(test)
reconstruc = gd.decode(enc)

pca = utils.compute_pca(patches, n_components=architecture[-1])
recont_pca = pca.transform(test)
recont_pca = pca.inverse_transform(recont_pca)

varrent = 0
varrentpca = 0
varrentcd1 = 0

for ii in range(np.shape(test)[0]):
	true = test[ii]
	approx = reconstruc[ii]
	approxcd1 = reconstruc[ii]
	approxpca = recont_pca[ii]
	
	div = np.linalg.norm(true)
	div = div * div
	
	norm = np.linalg.norm(true - approx)
	norm = norm * norm

	varrent += norm / div / np.shape(test)[0]
	
	norm = np.linalg.norm(true - approxpca)
	norm = norm * norm
	varrentpca += norm / div / np.shape(test)[0]
	
	norm = np.linalg.norm(true - approxcd1)
	norm = norm * norm
	varrentcd1 += norm / div / np.shape(test)[0]

print "Variance retention for gd:", varrent
print "Variance retention for cd1:", varrentcd1 
print "Variance retention for pca:", varrentpca

size = int(np.sqrt(np.shape(reconstruc)[1]))
# Show the original image, the reconstruction and the residues for the first 10 cases
for ii in range(10):
	img = test[ii].reshape(size,size)
	recon = reconstruc[ii].reshape(size,size)
	recont = recont_pca[ii].reshape(size,size)
	recon_cd1 = reconstruc[ii].reshape(size,size)

	plt.figure(figsize=(9,9))
	
	plt.subplot(3, 3, 1)
	plt.imshow((img), interpolation="nearest")
	
	plt.subplot(3, 3, 2)
	plt.title("Gradient Desc.")
	plt.imshow((recon), interpolation="nearest")
	plt.subplot(3, 3, 3)
	plt.imshow((img - recon), interpolation="nearest")
	plt.title("Gradient Desc. residues")
	
	plt.subplot(3, 3, 5)
	plt.title("CD1")
	plt.imshow((recon_cd1), interpolation="nearest")
	plt.subplot(3, 3, 6)
	plt.imshow((img - recon_cd1), interpolation="nearest")
	plt.title("CD1 residues")
	
	plt.subplot(3, 3, 8)
	plt.imshow((recont), interpolation="nearest")
	plt.title("PCA")
	plt.subplot(3, 3, 9)
	plt.imshow((img - recont), interpolation="nearest")
	plt.title("PCA residues")
	
plt.show()

plt.show()
