import numpy as np
import pylab as plt
import pylae
import pylae.utils as utils
from pylae.utils_mnist import load_MNIST_images

network_name = 'wip/dA-test-corr'

images = load_MNIST_images('data/mnist/train-images.idx3-ubyte')
images = images.T
patches = images[0:5000]#[0:50000]
test = images[50000:]

dA = pylae.dA.AutoEncoder(network_name)

n_pca=8

architecture = [2000, n_pca]
layers_type = ["SIGMOID", "SIGMOID", "SIGMOID", "SIGMOID", "SIGMOID"]

architecture = [100, n_pca]
layers_type = ["SIGMOID", "SIGMOID", "SIGMOID"]

architecture = [128, 64, n_pca]
layers_type = ["SIGMOID", "SIGMOID", "SIGMOID", "SIGMOID"]

pre_train = False
train = True
cost_fct = 'cross-entropy'
corruption = patches#[0., 0.3]

if pre_train:
	dA.pre_train(patches, architecture, layers_type, iterations=1000, mini_batch=0, corruption=corruption)
	utils.writepickle(dA.rbms, "%s/dA/layers.pkl" % network_name)
else:
	rbms = utils.readpickle("%s/dA/layers.pkl" % network_name)
	dA.set_autoencoder(rbms)
	dA.is_pretrained = True

if train:
	#dA.sgd(patches, iterations=1000, learning_rate=0.02, initial_momentum=0.5, final_momentum=0.9, annealing=2000,  corruption=corruption)
	dA.fine_tune(patches, iterations=1000, regularisation=0.000, sparsity=0.0, beta=0., corruption=corruption, cost_fct=cost_fct)#regularisation=0.000, sparsity=0.05, beta=3.)

	dA.save()
else:
	dA = utils.readpickle("%s/dA/ae.pkl" % network_name)
	print 'AE loaded!'
	
	
dA.display_train_history()
enc = dA.encode(test)

"""
print np.amin(enc[0]), np.amax(enc[0])
bins = np.logspace(np.log10(np.amin(enc[0])), np.log10(np.amax(enc[0])), 100, base=10)

import copy
coeffs = copy.deepcopy(enc[0])
coeffs.sort()
for c in coeffs:
	print c
print '----------------------------'
bins = range(1025)
ai = []
n_ae = 100

for ii in range(enc.shape[0]):
	#plt.figure()
	#plt.imshow(enc[ii].reshape(32,32), interpolation='nearest')
	cc = copy.deepcopy(enc[ii])
	ics = cc.argsort()
	ics = ics[::-1][:n_ae]
	ai.append(ics)
	#cc = cc[::-1]
	#print ics
	#print ii, len(enc[ii, enc[ii] > 0.65])
	#plt.show()

hist, bins = np.histogram(ai, bins)
for b, h in zip(bins, hist):
	print b, h 

hist = np.asarray(hist, dtype=np.float)
hist /= enc.shape[0]

print np.shape(hist)
print np.shape(bins)
plt.figure()
plt.bar(range(1024), hist)
plt.xlim([0,1024])

plt.figure()
hist.sort()
plt.plot(hist[::-1])
plt.axvline(n_ae, c='r')
plt.xlim([0, 1024])
plt.show()

#print enc.shape
#enc[enc < 0.8] = 0.
#print enc.shape

plt.figure()
plt.hist(enc[0], bins=bins)
plt.gca().set_xscale("log")
plt.show()
"""

reconstruc = dA.decode(enc)

pca = utils.compute_pca(patches, n_components=n_pca)
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
	recont = recont_pca[ii].reshape(size,size)
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
		
	plt.subplot(2, 3, 5)
	plt.imshow((recont), interpolation="nearest")
	plt.title("PCA")
	plt.subplot(2, 3, 6)
	plt.imshow((img - recont), interpolation="nearest")
	plt.title("PCA residues")

plt.show()
