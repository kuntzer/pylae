import numpy as np
import pylae.utils as utils
import galsim
import os
from scipy import stats

def get_shape(img):
	
	gps = galsim.Image(img)

	try:
		res = galsim.hsm.FindAdaptiveMom(gps)
		g1=res.observed_shape.g1
		g2=res.observed_shape.g2
		sigma=res.moments_sigma
		x = res.moments_centroid.x
		y = res.moments_centroid.y
	except:
		g1 = np.nan
		g2 = np.nan
		sigma = np.nan
		x = np.nan
		y = np.nan

	return g1, g2, x, y, sigma

def measure_stamps(dataset):
	
	measurments = []
	
	if not len(np.shape(dataset)) == 3:
		print 'WARNING: dataset has wrong number of dimensions, trying to get it right'
		m, npx = dataset.shape
		nsize = np.int(np.sqrt(npx))
		dataset = np.reshape(dataset, [m, nsize, nsize])
	
	for img in dataset:
		g1, g2, x, y, sigma = get_shape(img)
		noise = utils.skystats(img)['mad']
		measurments.append([g1, g2, sigma, noise, x, y])
	measurments = np.asarray(measurments)	
	
	return measurments

def analysis_plots(datasets, names=None, outdir='.', do_meas_params=True):
	import pylab as plt
	
	if names is None:
		names = ["signal", "dataset", "ae_reconstr", "pca_reconstr"]
	meas_params = {}
	
	fname_out_meas_params = os.path.join(outdir, 'meas_params.pkl')
	if do_meas_params:
		for dataset, name in zip(datasets, names):
			print 'Measuring %s...' % name, 
			rstamps = dataset.reshape(np.shape(dataset)[0], 24, 24)
			measurements = measure_stamps(rstamps)
			
			meas_params[name] = measurements
			print 'done.'
			
		utils.writepickle(meas_params, fname_out_meas_params)
	else:
		meas_params = utils.readpickle(fname_out_meas_params)
		
	e_sig = np.sqrt(meas_params["signal"][:,0]**2 + meas_params["signal"][:,1]**2) / 2.
	
	for name in names:
		es = np.sqrt(meas_params[name][:,0]**2 + meas_params[name][:,1]**2) / 2.
		print 'RMSD e', name, ':\t', utils.rmsd(es, e_sig),
		print '\tRMSD r', name, ':\t', utils.rmsd(meas_params[name][:,2], meas_params["signal"][:,2])/np.nanmean(meas_params["signal"][:,2])
		
	for name in names:
		es = np.sqrt(meas_params[name][:,0]**2 + meas_params[name][:,1]**2) / 2.
		print 'SEM e', name, ':\t', np.std(es - e_sig),
		print '\tSEM r', name, ':\t', np.std((meas_params[name][:,2] - meas_params["signal"][:,2])/np.nanmean(meas_params["signal"][:,2]))
	
	plt.figure(figsize=(16,12))
	plt.suptitle("ACTUAL SIGNAL")
	
	plt.subplot(241)
	plt.scatter(meas_params["signal"][:,0], meas_params["ae_reconstr"][:,0]-meas_params["signal"][:,0],)
	plt.subplot(242)
	plt.scatter(meas_params["signal"][:,1], meas_params["ae_reconstr"][:,1]-meas_params["signal"][:,1],)
	plt.subplot(243)
	plt.scatter(meas_params["signal"][:,2], meas_params["ae_reconstr"][:,2]-meas_params["signal"][:,2],)
	plt.subplot(244)
	plt.scatter(meas_params["signal"][:,3], meas_params["ae_reconstr"][:,3])
	plt.subplot(245)
	plt.scatter(meas_params["signal"][:,0], meas_params["pca_reconstr"][:,0]-meas_params["signal"][:,0],)
	plt.subplot(246)
	plt.scatter(meas_params["signal"][:,1], meas_params["pca_reconstr"][:,1]-meas_params["signal"][:,1],)
	plt.subplot(247)
	plt.scatter(meas_params["signal"][:,2], meas_params["pca_reconstr"][:,2]-meas_params["signal"][:,2],)
	plt.subplot(248)
	plt.scatter(meas_params["signal"][:,3], meas_params["pca_reconstr"][:,3])
	
	plt.figure(figsize=(16,12))
	plt.suptitle("NOISY SIGNAL")
	
	plt.subplot(241)
	plt.scatter(meas_params["dataset"][:,0], meas_params["ae_reconstr"][:,0]-meas_params["dataset"][:,0],)
	plt.subplot(242)
	plt.scatter(meas_params["dataset"][:,1], meas_params["ae_reconstr"][:,1]-meas_params["dataset"][:,1],)
	plt.subplot(243)
	plt.scatter(meas_params["dataset"][:,2], meas_params["ae_reconstr"][:,2]-meas_params["dataset"][:,2],)
	plt.subplot(244)
	plt.scatter(meas_params["dataset"][:,3], meas_params["ae_reconstr"][:,3])
	plt.subplot(245)
	plt.scatter(meas_params["dataset"][:,0], meas_params["pca_reconstr"][:,0]-meas_params["dataset"][:,0],)
	plt.subplot(246)
	plt.scatter(meas_params["dataset"][:,1], meas_params["pca_reconstr"][:,1]-meas_params["dataset"][:,1],)
	plt.subplot(247)
	plt.scatter(meas_params["dataset"][:,2], meas_params["pca_reconstr"][:,2]-meas_params["dataset"][:,2],)
	plt.subplot(248)
	plt.scatter(meas_params["dataset"][:,3], meas_params["pca_reconstr"][:,3])
	plt.show()
	
def reconstruction_plots(test_nonoise, test_dataset, test_tilde, test_pca_tilde):
	import pylab as plt
	vmin = np.amin(test_nonoise)
	vmax = np.amax(test_nonoise)
	for ii in range(10):
		residues = test_tilde[ii] - test_dataset[ii]
		
		f = plt.figure(figsize=(10,10))
		plt.subplot(3,4,1)
		plt.imshow(test_nonoise[ii].reshape(24,24), interpolation='None', vmin=vmin, vmax=vmax)
		plt.title('No noise img')
		
		plt.subplot(3,4,2)
		plt.imshow((test_dataset[ii]).reshape(24,24), interpolation='None', vmin=vmin, vmax=vmax)
		plt.title(r'$N(x)$')	
		
		stats.probplot((test_tilde[ii] - test_pca_tilde[ii]), plot=plt.subplot(3,4,3), dist='norm', fit=True)
		#stats.probplot(test_tilde[ii] - test_nonoise[ii], plot=plt.subplot(3,4,3), dist='norm', fit=True)
		plt.title("Normal Q-Q delta")
		
		stats.probplot(test_tilde[ii] - test_nonoise[ii], plot=plt.subplot(3,4,4), dist='norm', fit=True)
		plt.title("Normal Q-Q AE")
		
		plt.subplot(3,4,5)
		plt.imshow(test_tilde[ii].reshape(24,24), interpolation='None', vmin=vmin, vmax=vmax)
		plt.title('Reconstructed')
		
		plt.subplot(3,4,6)
		plt.imshow(residues.reshape(24,24), interpolation='None')
		plt.title(r'$\widetilde{N(x)} - N(x)$')
		
		plt.subplot(3,4,7)
		plt.imshow((test_tilde[ii] - test_nonoise[ii]).reshape(24,24), interpolation='None')#, vmin=vmin, vmax=vmax)
		plt.title(r'$\widetilde{N(x)} - x$')
		
		plt.subplot(3,4,8)
		plt.imshow((test_tilde[ii] - test_pca_tilde[ii]).reshape(24,24), interpolation='None')
		plt.title(r'$\widetilde{N(x)} - PCA({N(x)})$')
		
		plt.subplot(3,4,9)
		plt.imshow(test_pca_tilde[ii].reshape(24,24), interpolation='None', vmin=vmin, vmax=vmax)
		plt.title('PCA Reconstructed')
			
		plt.subplot(3,4,10)
		plt.title(r'$\Delta$')
		plt.imshow((test_pca_tilde[ii] - test_dataset[ii]).reshape(24,24), interpolation='None')
		
		plt.subplot(3,4,11)
		plt.imshow((test_pca_tilde[ii] - test_nonoise[ii]).reshape(24,24), interpolation='None')#, vmin=vmin, vmax=vmax)
		plt.title(r'$PCA({N(x)}) - x$')
		
		
		plt.subplot(3,4,12)
		plt.imshow((test_tilde[ii] / test_pca_tilde[ii]).reshape(24,24), interpolation='None')
		plt.title(r'$\widetilde{N(x)} / PCA({N(x)})$')

	plt.show()