import pylae
import pylae.utils as utils

import numpy as np
import pylab as plt
import os

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
datasize = 2000
trainper = 0.7
ind = np.int(trainper * datasize)

train_data = dataset[0:ind]
test_data = dataset[ind:datasize]
#test_data[0] = np.zeros_like(test_data[0])

print 'Shape of the training set: ', np.shape(train_data)
print 'Shape of the testing set: ', np.shape(test_data)


# Definition of the first half of the autoencoder -- the encoding bit.
# The deeper the architecture the more complex features can be learned.
architecture = [784, 392, 196, 98]
architecture = [256, 16]
# The layers_type must have len(architecture)+1 item.
# TODO: explain why and how to choose.
layers_type = ["SIGMOID", "SIGMOID", "SIGMOID", "SIGMOID", "LINEAR"]
layers_type = ["SIGMOID", "SIGMOID", "LINEAR"]

# Let's go
ae = pylae.autoencoder.AutoEncoder(network_name)
if pre_train:
	# This will train layer by layer the network by minimising the error
	# TODO: explain this in more details
	ae.pre_train(train_data, architecture, layers_type, learn_rate={'SIGMOID':0.1, 'LINEAR':1e-2}, 
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
	ae.backpropagation(train_data, iterations=500, learn_rate=0.1, momentum_rate=0.85)

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

# Build PCA:
pca = utils.compute_pca(train_data, n_components=architecture[-1])
recon_pca = pca.transform(train_data)
recon_pca = pca.inverse_transform(recon_pca)
rmsd_train_pca = utils.compute_rmsd(recon_pca, train_data) 

recont_pca = pca.transform(test_data)
recont_pca = pca.inverse_transform(recont_pca)
rmsd_test_pca = utils.compute_rmsd(recont_pca, test_data) 

# Compute the error on the ellipticity:
train_ell = []
recon_train_ell = []
test_ell = []
recon_test_ell = []
train_pca_ell = []
test_pca_ell = []
for ii in range(ind):
	img = train_data[ii].reshape(size,size)
	g1, g2 = utils.get_ell(img)
	train_ell.append([g1, g2, np.sqrt(g1*g1 + g2*g2)])
	
	img = reconstruc[ii].reshape(size,size)
	g1, g2 = utils.get_ell(img)
	recon_train_ell.append([g1, g2, np.sqrt(g1*g1 + g2*g2)])
	
	img = recon_pca[ii].reshape(size,size)
	g1, g2 = utils.get_ell(img)
	train_pca_ell.append([g1, g2, np.sqrt(g1*g1 + g2*g2)])
	
	if ii >= datasize - ind: continue
		
	img = test_data[ii].reshape(size,size)
	g1, g2 = utils.get_ell(img)
	test_ell.append([g1, g2, np.sqrt(g1*g1 + g2*g2)])
	
	img = reconstructest[ii].reshape(size,size)
	g1, g2 = utils.get_ell(img)
	recon_test_ell.append([g1, g2, np.sqrt(g1*g1 + g2*g2)])
	
	img = recont_pca[ii].reshape(size,size)
	g1, g2 = utils.get_ell(img)
	test_pca_ell.append([g1, g2, np.sqrt(g1*g1 + g2*g2)])
	
train_ell = np.asarray(train_ell)
recon_train_ell = np.asarray(recon_train_ell)
test_ell = np.asarray(test_ell)
recon_test_ell = np.asarray(recon_test_ell)
train_pca_ell = np.asarray(train_pca_ell)
test_pca_ell = np.asarray(test_pca_ell)


pca_ell_error = test_pca_ell[:,2] - test_ell[:,2]
rms_pca_err = np.sqrt(pca_ell_error*pca_ell_error)
ae_ell_error = recon_test_ell[:,2] - test_ell[:,2]
rms_ae_err = np.sqrt(ae_ell_error*ae_ell_error)

tru_ell = np.sqrt(truth[0:ind,0]*truth[0:ind,0] + truth[0:ind,1]*truth[0:ind,1])
gs_ell_error = train_ell[:,2] - tru_ell
gs_error = np.sqrt(np.mean(gs_ell_error * gs_ell_error))

pca_error = np.sqrt(np.mean(pca_ell_error * pca_ell_error))
ae_error = np.sqrt(np.mean(ae_ell_error * ae_ell_error))
gs_error = np.sqrt(np.mean(gs_error * gs_error))
print "ERROR ON TEST DATA"
print 'gs error :', gs_error
print 'ae error :', ae_error
print 'pca error:', pca_error

plt.figure()
plt.hist(rms_pca_err, label="PCA", alpha=0.5)
plt.hist(rms_ae_err, label="Auto-encoder", alpha=0.5)

plt.axvline(pca_error, color="blue", lw=2)
plt.axvline(ae_error, color="green", lw=2)
plt.xlabel("RMS(Model - Data)")
plt.legend(loc="best")
plt.grid()

#########################################################################

pca_ell_error = train_pca_ell[:,2] - tru_ell
rms_pca_err = np.sqrt(pca_ell_error*pca_ell_error)
ae_ell_error = recon_train_ell[:,2] - tru_ell
rms_ae_err = np.sqrt(ae_ell_error*ae_ell_error)

pca_error = np.sqrt(np.mean(pca_ell_error * pca_ell_error))
ae_error = np.sqrt(np.mean(ae_ell_error * ae_ell_error))
print 'ae absolute error :', ae_error
print 'pca absolute error:', pca_error

plt.figure()
plt.hist(rms_pca_err, label="PCA", alpha=0.5)
plt.hist(rms_ae_err, label="Auto-encoder", alpha=0.5)

plt.axvline(pca_error, color="blue", lw=2)
plt.axvline(ae_error, color="green", lw=2)
plt.xlabel("RMS(Model - True)")
plt.legend(loc="best")
plt.grid()
plt.show()


# Show the original test image, the reconstruction and the residues for the first 10 cases
for ii in range(10):
	img = test_data[ii].reshape(size,size)
	recon = reconstructest[ii].reshape(size,size)
	recont_pca_img = recont_pca[ii].reshape(size,size)

	try:
		dg1, dg2 = utils.get_ell(img)
	except :
		dg1 = 20
		dg2 = 20

	rg1, rg2, rg = recon_test_ell[ii]

	pg1, pg2 = utils.get_ell(recont_pca_img)
	pg1, pg2, pg = test_pca_ell[ii]
	
	tg1, tg2 = truth[ind+ii,0:2]
	
	plt.figure()
	
	plt.subplot(1, 3, 1)
	plt.imshow((img), interpolation="nearest")
	plt.text(0,size*1.2,"g1 = %1.4f\ng2 = %1.4f\ng=%1.4f\n...........\ntg1 = %1.4f\ntg2 = %1.4f\ng=%1.4f" % (
		dg1, dg2, np.hypot(dg1, dg2), tg1, tg2, np.hypot(tg1, tg2)), va="top")
	plt.title("Data")
	plt.subplot(1, 3, 2)
	plt.imshow((recon), interpolation="nearest")
	plt.text(0,size*1.2,"g1 = %1.4f\ng2 = %1.4f\ng=%1.4f" % (rg1, rg2, np.hypot(rg1, rg2)), va="top")
	#plt.colorbar()
	plt.title("Auto-encoder")
	plt.subplot(1, 3, 3)
	#plt.imshow((img - recon), interpolation="nearest")
	plt.imshow(recont_pca_img, interpolation="nearest")
	plt.text(0,size*1.2,"g1 = %1.4f\ng2 = %1.4f\ng = %1.4f" % (pg1, pg2, np.hypot(pg1, pg2)), va="top")
	#plt.colorbar()
	plt.title("PCA")
	
	"""
	#plt.subplot(2, 3, 4)
	#plt.imshow((corr), interpolation="nearest")
	#plt.colorbar()
	plt.subplot(2, 3, 5)
	plt.title("Residues AE")
	plt.imshow((recon - img), interpolation="nearest")
	#plt.colorbar()
	plt.subplot(2, 3, 6)
	plt.title("Residues PCA")
	plt.imshow((recont_pca_img-img), interpolation="nearest")
	#plt.colorbar()
	plt.xlabel("RMS Deviation : %1.4f" % (rmsd_test[ii]))
	"""
plt.show()