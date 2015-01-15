import pylae
import pylae.utils as utils

import numpy as np
import pylab as plt
import os		
		
import galsim

network_name = "gauss_psf_"

# Load the data and remember the size of the data (assume a square image here)
# ! The data *must* be normalised here
#dataset = np.loadtxt("data/psfs-gs_rnd.dat", delimiter=",")
dataset = np.loadtxt("data/psfs-smalldev.dat", delimiter=",")

dataset, low, high = utils.normalise(dataset)
size = np.sqrt(np.shape(dataset)[1])

# Can we skip some part of the training ?
pre_train = False
train = pre_train

# Separate into training and testing data
datasize = np.shape(dataset)[0]
trainper = 0.7
ind = np.int(trainper * datasize)

train_data = dataset[0:ind]
test_data = dataset[ind:]
test_data = train_data
#test_data[0] = np.zeros_like(test_data[0])

print 'Shape of the training set: ', np.shape(train_data)
print 'Shape of the testing set: ', np.shape(test_data)

# Definition of the first half of the autoencoder -- the encoding bit.
# The deeper the architecture the more complex features can be learned.
architecture = [784, 392, 196, 98, 16]
#architecture = [256, 128, 64, 32]
# The layers_type must have len(architecture)+1 item.
# TODO: explain why and how to choose.
layers_type = ["SIGMOID", "SIGMOID", "SIGMOID", "SIGMOID", "SIGMOID", "SIGMOID"]

# Let's go
ae = pylae.autoencoder.AutoEncoder(network_name)
if pre_train:
	# This will train layer by layer the network by minimising the error
	# TODO: explain this in more details
	ae.pre_train(train_data, architecture, layers_type, learn_rate={'SIGMOID':3e-3, 'LINEAR':3e-4}, 
				iterations=2000, mini_batch=100)
	
	# Save the resulting layers
	utils.writepickle(ae.rbms, "%srbms.pkl" % network_name)
	
elif not pre_train and train :
	rbms = utils.readpickle("%srbms.pkl" % network_name)
	# An autoencoder instance was created some lines earlier, preparing the other half of the 
	# network based on layers loaded from the pickle file. 
	ae.set_autencoder(rbms)
	
if train:
	print 'Starting backpropagation'
	ae.backpropagation(train_data, iterations=150, learn_rate=0.001, momentum_rate=0.9)

	ae.save("%sautoencoder.pkl" % network_name)
	
	os.system("/usr/bin/canberra-gtk-play --id='complete-media-burn'")

else :
	ae = utils.readpickle("%sautoencoder.pkl" % network_name)

# Use the training data as if it were a training set
reconstruc = ae.feedforward(train_data)
#np.random.shuffle(test_data)
reconstructest = ae.feedforward(test_data)

# Compute the RMSD error for the training set
rmsd_train = utils.compute_rmsd(train_data, reconstruc)
recon_avg = np.sum(train_data - reconstruc, axis=0)
recon_avg /= np.shape(train_data)[0]
corr = recon_avg.reshape(size,size)

# Compute the RMSD error for the test set
rmsd_test = utils.compute_rmsd(test_data, reconstructest)

# Show the figures for the distribution of the RMSD and the learning curves
plt.figure()
pylae.plots.hist(rmsd_train, rmsd_test)

ae.plot_rmsd_history()

truth = np.loadtxt("data/truth-smalldev.dat", delimiter=",")

# Show the original test image, the reconstruction and the residues for the first 10 cases
for ii in range(10):
	img = test_data[ii].reshape(size,size)
	recon = reconstructest[ii].reshape(size,size)

	try:
		gps = galsim.Image(img)
		res = galsim.hsm.FindAdaptiveMom(gps)
		dg1=res.observed_shape.g1
		dg2=res.observed_shape.g2
	except :
		dg1 = 20
		dg2 = 20
		
	gps = galsim.Image(recon)
	res = galsim.hsm.FindAdaptiveMom(gps)
	rg1=res.observed_shape.g1
	rg2=res.observed_shape.g2

	tg1, tg2 = truth[ii,0:2]
	
	plt.figure()
	
	plt.subplot(1, 3, 1)
	plt.imshow((img), interpolation="nearest")
	plt.text(0,size*1.2,"g1 = %1.4f\ng2 = %1.4f\n...........\ntg1 = %1.4f\ntg2 = %1.4f" % (dg1, dg2, tg1, tg2), va="top")
	plt.subplot(1, 3, 2)
	plt.imshow((recon), interpolation="nearest")
	plt.text(0,size*1.2,"g1 = %1.4f\ng2 = %1.4f" % (rg1, rg2), va="top")
	#plt.colorbar()
	plt.subplot(1, 3, 3)
	plt.imshow((img - recon), interpolation="nearest")
	plt.colorbar()
	
	
	"""
	plt.colorbar()
	plt.subplot(2, 3, 4)
	plt.imshow((corr), interpolation="nearest")
	#plt.colorbar()
	plt.subplot(2, 3, 5)
	plt.imshow((recon + corr), interpolation="nearest")
	plt.colorbar()
	plt.subplot(2, 3, 6)
	plt.imshow((img - corr - recon), interpolation="nearest")
	plt.colorbar()
	plt.xlabel("RMS Deviation : %1.4f" % (rmsd_test[ii]))
	"""
plt.show()