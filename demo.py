import pylae
import pylae.utils as utils

import numpy as np
import pylab as plt
import os	
import copy	
		
network_name = "demo_"

# Load the data and remember the size of the data (assume a square image here)
# ! The data *must* be normalised here
data = np.loadtxt("data/digit.dat", delimiter=",")
data, _ , _ = utils.normalise(data)
datall = data
#data = data.T
data = data[:700]
size = np.sqrt(np.shape(data)[1])

# Can we skip some part of the training ?
pre_train = False
train = True#pre_train

# Definition of the first half of the autoencoder -- the encoding bit.
# The deeper the architecture the more complex features can be learned.
architecture = [144, 64, 9]
# The layers_type must have len(architecture)+1 item.
# TODO: explain why and how to choose.
layers_type = ["SIGMOID", "SIGMOID", "SIGMOID", "LINEAR"]

# Let's go
gd = pylae.autoencoder.AutoEncoder('demo_gradientdescent', rbm_type="gd")
cd1 = pylae.autoencoder.AutoEncoder('demo_gradientdescent', rbm_type="cd1")
if pre_train:
	# This will train layer by layer the network by minimising the error
	# TODO: explain this in more details
	gd.pre_train(data, architecture, layers_type, learn_rate={'SIGMOID':0.1, 'LINEAR':0.01}, 
				iterations=2000, mini_batch=100)
	
	# Save the resulting layers
	utils.writepickle(gd.rbms, "%sgdrbms.pkl" % network_name)
	
	# This will train layer by layer the network by minimising the error
	# TODO: explain this in more details
	cd1.pre_train(data, architecture, layers_type, learn_rate={'SIGMOID':0.1, 'LINEAR':0.01}, 
				iterations=2000, mini_batch=100)
	
	
	# Save the resulting layers
	utils.writepickle(cd1.rbms, "%scd1rbms.pkl" % network_name)
	
elif not pre_train and train :
	rbms = utils.readpickle("%sgdrbms.pkl" % network_name)
	# An autoencoder instance was created some lines earlier, preparing the other half of the 
	# network based on layers loaded from the pickle file. 
	gd.set_autoencoder(rbms)

	gd.is_pretrained = True
	
	rbms = utils.readpickle("%scd1rbms.pkl" % network_name)
	# An autoencoder instance was created some lines earlier, preparing the other half of the 
	# network based on layers loaded from the pickle file. 
	cd1.set_autoencoder(rbms)
	cd1.is_pretrained = True
	
if train:
	print 'Starting backpropagation'
	#gd.backpropagation(data, iterations=2000, learn_rate=0.2, momentum_rate=0.9, regularisation=0.0)
	gd.backpropagation(data, iterations=1000, learn_rate=0.2, momentum_rate=0.9, regularisation=0.001,
					sparsity=0.2)

	gd.save("%sgdautoencoder.pkl" % network_name)
	
	#cd1.backpropagation(data, iterations=2000, learn_rate=0.2, momentum_rate=0.9, regularisation=0.001,
	#				sparsity=0.2)

	#cd1.save("%scd1autoencoder.pkl" % network_name)
	
	os.system("/usr/bin/canberra-gtk-play --id='complete-media-burn'")

else :
	gd = utils.readpickle("%s%s/gd/ae.pkl" % (network_name, "gradientdescent"))
	cd1 = utils.readpickle("%s%s/cd1/ae.pkl" % (network_name, "gradientdescent"))

# Use the training data as if it were a training set
test = copy.deepcopy(datall[700:,:])
np.random.shuffle(test)

reconstruc = gd.feedforward(test)
gd.visualise(0)

reconstruc_cd1 = cd1.feedforward(test)

# Compute the RMSD error for the training set
rmsd = utils.compute_rmsd(test, reconstruc)

# Show the figures for the distribution of the RMSD and the learning curves
plt.figure()
plt.hist(rmsd)

"""
gd.plot_rmsd_history()
cd1.plot_rmsd_history()
"""

pca = utils.compute_pca(data, n_components=architecture[-1])
recont_pca = pca.transform(test)
recont_pca = pca.inverse_transform(recont_pca)

varrent = 0
varrentpca = 0
varrentcd1 = 0

for ii in range(np.shape(test)[0]):
	true = test[ii]
	approx = reconstruc[ii]
	approxcd1 = reconstruc_cd1[ii]
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

# Show the original image, the reconstruction and the residues for the first 10 cases
for ii in range(20):
	img = test[ii].reshape(size,size)
	recon = reconstruc[ii].reshape(size,size)
	recont = recont_pca[ii].reshape(size,size)
	recon_cd1 = reconstruc_cd1[ii].reshape(size,size)

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
