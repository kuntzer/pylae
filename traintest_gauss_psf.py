import pylae
import pylae.utils as utils

import numpy as np
import pylab as plt
import os

run_name = 'smalldev-noisy'
network_name = run_name
# Load the data and remember the size of the data (assume a square image here)
# ! The data *must* be normalised here
#dataset = np.loadtxt("data/psfs-gs_rnd.dat", delimiter=",")
dataset = np.loadtxt("data/psfs-%s.dat" % run_name, delimiter=",")

dataset, low, high = utils.normalise(dataset)
size = np.sqrt(np.shape(dataset)[1])

truthset = np.loadtxt("data/psfs-true-%s.dat" % run_name, delimiter=",")
truthset, _, _ = utils.normalise(truthset)

truth = np.loadtxt("data/truth-%s.dat" % run_name, delimiter=",")


# Can we skip some part of the training ?
pre_train = False
train = False

# Add the mean of the residues ?
do_corr = False

fancy = False

# Definition of the first half of the autoencoder -- the encoding bit.
# The deeper the architecture the more complex features can be learned.
architecture = [2000, 1000, 500, 100, 15]
architecture = [960, 480, 240, 120, 60, 30, 15]
#architecture = [4287, 15]
architecture = [700, 300, 128]
#architecture = [128, 15]
pca_nb_components = architecture[-1]

# The layers_type must have len(architecture)+1 item.
# TODO: explain why and how to choose.
layers_type = ["SIGMOID", "SIGMOID", "SIGMOID", "SIGMOID", "SIGMOID", "LINEAR"]
layers_type = ["SIGMOID", "SIGMOID", "SIGMOID", "SIGMOID", "SIGMOID", "SIGMOID", "SIGMOID", "LINEAR"]
#layers_type = ["SIGMOID", "SIGMOID", "LINEAR"]
layers_type = ["SIGMOID", "SIGMOID", "SIGMOID", "LINEAR"]
# Separate into training and testing data
datasize = np.shape(dataset)[0]
datasize = 2000#10000
trainper = 0.7
ind = np.int(trainper * datasize)

train_data = dataset[0:ind]
test_data = dataset[ind:datasize]

train_true = truthset[0:ind]
test_true = truthset[ind:datasize]
#test_data[0] = np.zeros_like(test_data[0])

print 'Shape of the training set: ', np.shape(train_data)
print 'Shape of the testing set: ', np.shape(test_data)

# Let's go
ae = pylae.autoencoder.AutoEncoder(network_name, rbm_type="gd")
if pre_train:
	# This will train layer by layer the network by minimising the error
	# TODO: explain this in more details
	ae.pre_train(train_data, architecture, layers_type, learn_rate={'SIGMOID':0.034, 'LINEAR':0.034/10.}, 
				initialmomentum=0.53,finalmomentum=0.93, iterations=2000, mini_batch=100, regularisation=0.002)
	
	# Save the resulting layers
	utils.writepickle(ae.rbms, "%srbms.pkl" % network_name)
	
elif not pre_train and train :
	rbms = utils.readpickle("%srbms.pkl" % network_name)
	# An autoencoder instance was created some lines earlier, preparing the other half of the 
	# network based on layers loaded from the pickle file. 
	ae.set_autencoder(rbms)
	ae.is_pretrained = True
	
if train:
	print 'Starting backpropagation'
	ae.backpropagation(train_data, iterations=1000, learn_rate=0.13, momentum_rate=0.83)

	ae.save("%sautoencoder.pkl" % network_name)
	
	os.system("/usr/bin/canberra-gtk-play --id='complete-media-burn'")

else :
	ae = utils.readpickle("%sautoencoder.pkl" % network_name)

print "Training complete."
# Use the training data as if it were a training set
reconstruc = ae.feedforward(train_data)
#np.random.shuffle(test_data)
reconstructest = ae.feedforward(test_data)
print "Reconstruction complete"

# Compute the RMSD error for the training set
recon_avg = np.mean(train_data - reconstruc, axis=0)
#recon_avg /= np.shape(train_data)[0]
if do_corr:
	reconstruc += recon_avg
	reconstructest += recon_avg

corr = recon_avg.reshape(size,size)
rmsd_train = utils.compute_rmsd(reconstruc, train_true)

# Compute the RMSD error for the test set
rmsd_test = utils.compute_rmsd(reconstructest, test_true)

truth_train = truth[0:ind]
truth_test = truth[ind:datasize]

# Build PCA:
if train :
	pca = utils.compute_pca(train_data, n_components=pca_nb_components)
	utils.writepickle(pca, "%spca.pkl" % network_name)
else: 
	pca = utils.readpickle("%spca.pkl" % network_name)
	
recon_pca = pca.transform(train_data)
recon_pca = pca.inverse_transform(recon_pca)
rmsd_train_pca = utils.compute_rmsd(recon_pca, train_true) 

recont_pca = pca.transform(test_data)
recont_pca = pca.inverse_transform(recont_pca)
rmsd_test_pca = utils.compute_rmsd(recont_pca, test_true) 

print "RMSD ERRORS ON IMAGES"
print "TRAIN"
print "ae :", np.mean(rmsd_train)
print "pca:", np.mean(rmsd_train_pca)
print "TEST"
print "ae :", np.mean(rmsd_test)
print "pca:", np.mean(rmsd_test_pca)

# Compute the error on the ellipticity:
train_ell = []
recon_train_ell = []
test_ell = []
recon_test_ell = []
train_pca_ell = []
test_pca_ell = []
noise_train = []
noise_test = []
noise_test_pca = []
for ii in range(ind):
	img = train_data[ii].reshape(size,size)
	g1, g2 = utils.get_ell(img)
	train_ell.append([g1, g2, np.sqrt(g1*g1 + g2*g2)])
	
	img = reconstruc[ii].reshape(size,size) + corr
	g1, g2 = utils.get_ell(img)
	recon_train_ell.append([g1, g2, np.sqrt(g1*g1 + g2*g2)])
	noise_train.append(utils.skystats(img)['mad'])
	
	img = recon_pca[ii].reshape(size,size)
	g1, g2 = utils.get_ell(img)
	train_pca_ell.append([g1, g2, np.sqrt(g1*g1 + g2*g2)])
	
	if ii >= datasize - ind: continue
		
	img = test_data[ii].reshape(size,size)
	g1, g2 = utils.get_ell(img)
	test_ell.append([g1, g2, np.sqrt(g1*g1 + g2*g2)])
	
	img = reconstructest[ii].reshape(size,size) + corr
	g1, g2 = utils.get_ell(img)
	recon_test_ell.append([g1, g2, np.sqrt(g1*g1 + g2*g2)])
	noise_test.append(utils.skystats(img)['mad'])
	
	img = recont_pca[ii].reshape(size,size)
	g1, g2 = utils.get_ell(img)
	test_pca_ell.append([g1, g2, np.sqrt(g1*g1 + g2*g2)])
	noise_test_pca.append(utils.skystats(img)['mad'])
	
train_ell = np.asarray(train_ell)
recon_train_ell = np.asarray(recon_train_ell)
test_ell = np.asarray(test_ell)
recon_test_ell = np.asarray(recon_test_ell)
train_pca_ell = np.asarray(train_pca_ell)
test_pca_ell = np.asarray(test_pca_ell)

print "ELL ERROR ON TRAIN DATA"
pca_ell_error = train_pca_ell[:,2] - train_ell[:,2]
rms_pca_err = np.sqrt(pca_ell_error*pca_ell_error)
ae_ell_error = recon_train_ell[:,2] - train_ell[:,2]
rms_ae_err = np.sqrt(ae_ell_error*ae_ell_error)

tru_ell = np.sqrt(truth[0:ind,0]*truth[0:ind,0] + truth[0:ind,1]*truth[0:ind,1])
gs_ell_error = train_ell[:,2] - tru_ell
gs_error = np.sqrt(np.mean(gs_ell_error * gs_ell_error))

pca_error = np.sqrt(np.mean(pca_ell_error * pca_ell_error))
ae_error = np.sqrt(np.mean(ae_ell_error * ae_ell_error))
gs_error = np.sqrt(np.mean(gs_error * gs_error))
print 'gs error :', gs_error
print 'ae error :', ae_error
print 'pca error:', pca_error

print "ELL ERROR ON TEST DATA"
pca_ell_error = test_pca_ell[:,2] - test_ell[:,2]
rms_pca_err = np.sqrt(pca_ell_error*pca_ell_error)
ae_ell_error = recon_test_ell[:,2] - test_ell[:,2]
rms_ae_err = np.sqrt(ae_ell_error*ae_ell_error)

tru_ell = np.sqrt(truth[ind:datasize,0]*truth[ind:datasize,0] + truth[ind:datasize,1]*truth[ind:datasize,1])
gs_ell_error = test_ell[:,2] - tru_ell
gs_error = np.sqrt(np.mean(gs_ell_error * gs_ell_error))

pca_error = np.sqrt(np.mean(pca_ell_error * pca_ell_error))
ae_error = np.sqrt(np.mean(ae_ell_error * ae_ell_error))
gs_error = np.sqrt(np.mean(gs_error * gs_error))
print 'gs error :', gs_error
print 'ae error :', ae_error
print 'pca error:', pca_error

# Show the figures for the distribution of the RMSD and the learning curves
if fancy : pylae.figures.set_fancy()

plt.figure()
pylae.plots.hist(rmsd_train, rmsd_test, rmsd_test_pca, xlabel="RMSD")
plt.figure()
pylae.plots.hist(noise_train, noise_test, noise_test_pca, xlabel="Noise")

ae.plot_rmsd_history()

plt.figure()
plt.scatter(test_pca_ell[:,2], rmsd_test_pca, label="PCA", color="b")
plt.scatter(recon_test_ell[:,2], rmsd_test, label="Auto-encoder", color="g")
pylae.plots.hspans(rmsd_test, "g")
pylae.plots.hspans(rmsd_test_pca, "b")

plt.xlim([0, 1.02*np.amax([np.amax(recon_test_ell[:,2]), np.amax(test_pca_ell[:,2])])])
plt.ylim([0.98*np.amin([np.amin(rmsd_test), np.amin(rmsd_test_pca)]),\
	1.02*np.amax([np.amax(rmsd_test), np.amax(rmsd_test_pca)])])

plt.xlabel("e of reconstructed data")
plt.ylabel("RMSD error")
plt.legend(loc="best")
plt.grid()

plt.figure()
plt.scatter(test_pca_ell[:,2], noise_test_pca, label="PCA", color="b")
plt.scatter(recon_test_ell[:,2], noise_test, label="Auto-encoder", color="g")
pylae.plots.hspans(noise_test, "g")
pylae.plots.hspans(noise_test_pca, "b")

plt.xlim([0, 1.02*np.amax([np.amax(recon_test_ell[:,2]), np.amax(test_pca_ell[:,2])])])
plt.ylim([0.98*np.amin([np.amin(noise_test), np.amin(noise_test_pca)]),\
	1.02*np.amax([np.amax(noise_test), np.amax(noise_test_pca)])])

plt.xlabel("Ellipticity of reconstructed data")
plt.ylabel("Noise")
plt.legend(loc="best")
plt.grid()

plt.figure()
plt.scatter(tru_ell, test_pca_ell[:,2], label="PCA", color="b")
plt.scatter(tru_ell, recon_test_ell[:,2], label="Auto-encoder", color="g")
#pylae.plots.hspans(noise_test, "g")
#pylae.plots.hspans(noise_test_pca, "b")
plt.plot([0.,np.amax(tru_ell)], [0.,np.amax(tru_ell)], 'r--', lw=2)
plt.xlim([0, 1.02*np.amax(tru_ell)])
plt.ylim([0.98*np.amin([np.amin(test_pca_ell[:,2]), np.amin(recon_test_ell[:,2])]),\
	1.02*np.amax([np.amax(test_pca_ell[:,2]), np.amax(recon_test_ell[:,2])])])

plt.xlabel("Ellipticity of true data")
plt.ylabel("Ellipticity of reconstructed data")
plt.legend(loc="best")
plt.grid()

plt.figure()
plt.hist(rms_pca_err, label="PCA", alpha=0.5)
plt.hist(rms_ae_err, label="Auto-encoder", alpha=0.5)

plt.axvline(pca_error, color="blue", lw=2)
plt.axvline(ae_error, color="green", lw=2)
plt.xlabel("RMS(Model - Data)")
plt.legend(loc="best")
plt.grid()

#########################################################################

pca_ell_error = test_pca_ell[:,2] - tru_ell
rms_pca_err = np.sqrt(pca_ell_error*pca_ell_error)
ae_ell_error = recon_test_ell[:,2] - tru_ell
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
	recon = reconstructest[ii].reshape(size,size)# + corr
	recont_pca_img = recont_pca[ii].reshape(size,size)
	
	truimg = truthset[ind+ii].reshape(size,size)
	"""
	img = train_data[ii].reshape(size,size)
	recon = reconstruc[ii].reshape(size,size) + corr
	recont_pca_img = recon_pca[ii].reshape(size,size)
	"""
	try:
		dg1, dg2 = utils.get_ell(img)
	except :
		dg1 = 20
		dg2 = 20
	dn = utils.skystats(img)["mad"]

	rg1, rg2, rg = recon_test_ell[ii]
	rn = utils.skystats(recon)["mad"]

	pg1, pg2 = utils.get_ell(recont_pca_img)
	pg1, pg2, pg = test_pca_ell[ii]
	pn = utils.skystats(recont_pca_img)["mad"]
	
	tg1, tg2 = truth[ind+ii,0:2]
	
	plt.figure()
	
	plt.subplot(2, 3, 1)
	plt.imshow((img), interpolation="nearest")
	plt.text(0,size*1.2,"g1 = %1.4f\ng2 = %1.4f\ng=%1.4f\n\nnoise = %1.4f\n...........\ntg1 = %1.4f\ntg2 = %1.4f\ng=%1.4f" % (
		dg1, dg2, np.hypot(dg1, dg2), dn, tg1, tg2, np.hypot(tg1, tg2)), va="top")
	plt.title("Data")
	plt.subplot(2, 3, 2)
	plt.imshow((recon), interpolation="nearest")
	plt.text(0,size*1.2,"g1 = %1.4f\ng2 = %1.4f\ng=%1.4f\n\nnoise = %1.4f" % (rg1, rg2, np.hypot(rg1, rg2), rn), va="top")
	#plt.colorbar()
	plt.title("Auto-encoder")
	plt.subplot(2, 3, 3)
	#plt.imshow((img - recon), interpolation="nearest")
	plt.imshow(recont_pca_img, interpolation="nearest")
	plt.text(0,size*1.2,"g1 = %1.4f\ng2 = %1.4f\ng = %1.4f\n\nnoise = %1.4f" % (pg1, pg2, np.hypot(pg1, pg2), pn), va="top")
	#plt.colorbar()
	plt.title("PCA")
	
	plt.subplot(2, 3, 4)
	plt.title("data - true")
	plt.imshow((img-truimg), interpolation="nearest")
	
	plt.subplot(2, 3, 5)
	plt.title("Residues AE")
	plt.imshow((recon - img), interpolation="nearest")
	plt.colorbar()
	plt.subplot(2, 3, 6)
	plt.title("Residues PCA")
	plt.imshow((recont_pca_img-img), interpolation="nearest")
	plt.colorbar()
	plt.xlabel("RMS Deviation : %1.4f" % (rmsd_test[ii]))
	
	plt.subplots_adjust(hspace = 1.8)
plt.show()
