import numpy as np
import pylae.utils as utils
import galsim
import os
from scipy import stats
import scipy

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
	
	ii = 0
	for img in dataset:
		g1, g2, x, y, sigma = get_shape(img)
		noise = utils.skystats(img)['mad']
		measurments.append([g1, g2, sigma, noise, x, y])
		if np.any(np.isnan(measurments)) and False:# or ii > 186 :
			print 'NAN ALERT!'
			print ii
			print g1, g2, sigma, noise, x, y
			print np.amin(img), np.amax(img)
			import pylab as plt
			npix = int(np.sqrt(img.size))
			plt.figure()
			plt.imshow(img.reshape(npix,npix), interpolation="None")
			plt.title(ii)
			plt.show()
		ii+=1
			
	measurments = np.asarray(measurments)	
	
	return measurments

def analysis_plots(datasets, parameters=None, names=None, outdir='.', do_meas_params=True):
	import pylab as plt
	
	if names is None:
		names = ["signal", "dataset", "ae_reconstr", "pca_reconstr"]
	meas_params = {}
	
	fname_out_meas_params = os.path.join(outdir, 'meas_params.pkl')
	
	if parameters is None:
		parameters = ['e1', 'e2', 'dr2']
	
	if do_meas_params:
		for dataset, name in zip(datasets, names):
			npix = int(np.sqrt(np.shape(dataset)[1]))
			print 'Measuring %s...' % name, 
			rstamps = dataset.reshape(np.shape(dataset)[0], npix, npix)
			measurements = measure_stamps(rstamps)
			
			meas_params[name] = measurements
			print 'done.'
			
		utils.writepickle(meas_params, fname_out_meas_params)
	else:
		meas_params = utils.readpickle(fname_out_meas_params)
		
	e_sig = np.sqrt(meas_params["signal"][:,0]**2 + meas_params["signal"][:,1]**2) / 2.
	
	for name in names:
		es = np.sqrt(meas_params[name][:,0]**2 + meas_params[name][:,1]**2) / 2.
		print 'RMSD e', name, ':\t','%1.3e' % utils.rmsd(es, e_sig),
		print '\tRMSD r', name, ':\t', '%1.3e' % (utils.rmsd(meas_params[name][:,2], meas_params["signal"][:,2])/np.nanmean(meas_params["signal"][:,2]))
		print 'mean e1', name, ':\t', ('%1.3e%+1.3e' % (np.mean(meas_params[name][:,0]-meas_params["signal"][:,0]),np.std(meas_params[name][:,0]))),
		print 'mean e2', name, ':\t', ('%1.3e%+1.3e' % (np.mean(meas_params[name][:,1]-meas_params["signal"][:,1]),np.std(meas_params[name][:,1])))
		print 'mean dr', name, ':\t', ('%1.3e%+1.3e' % (np.mean(meas_params[name][:,2]-meas_params["signal"][:,2]),np.std(meas_params[name][:,2]))),
		print 'median dr', name, ':\t', ('%1.3e%+1.3e' % (np.median(meas_params[name][:,2]-meas_params["signal"][:,2]),np.std(meas_params[name][:,2])))
		
		for idd, npa in enumerate(parameters):
			(m, c) = scipy.polyfit(meas_params["signal"][:,idd],meas_params[name][:,idd]-meas_params["signal"][:,idd],1)
			print 'bias %s : m=%1.3e, c=%1.3e' % (npa, m, c) 
		print 
		
	for name in names:
		es = np.sqrt(meas_params[name][:,0]**2 + meas_params[name][:,1]**2) / 2.
		print 'SEM e', name, ':\t', '%1.3e' %np.std(es - e_sig),
		print '\tSEM r', name, ':\t', '%1.3e' % (np.std((meas_params[name][:,2] - meas_params["signal"][:,2])/np.nanmean(meas_params["signal"][:,2])))
	
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
	npix = int(np.sqrt(np.shape(test_nonoise)[1]))
	
	for ii in range(10):
		residues = test_tilde[ii] - test_dataset[ii]
		
		f = plt.figure(figsize=(10,10))
		plt.subplot(3,4,1)
		plt.imshow(test_nonoise[ii].reshape(npix,npix), interpolation='None', vmin=vmin, vmax=vmax)
		plt.title('No noise img')
		
		plt.subplot(3,4,2)
		plt.imshow((test_dataset[ii]).reshape(npix,npix), interpolation='None', vmin=vmin, vmax=vmax)
		plt.title(r'$N(x)$')	
		
		stats.probplot((test_tilde[ii] - test_pca_tilde[ii]), plot=plt.subplot(3,4,3), dist='norm', fit=True)
		#stats.probplot(test_tilde[ii] - test_nonoise[ii], plot=plt.subplot(3,4,3), dist='norm', fit=True)
		plt.title("Normal Q-Q PCA")
		
		stats.probplot(test_tilde[ii] - test_nonoise[ii], plot=plt.subplot(3,4,4), dist='norm', fit=True)
		plt.title("Normal Q-Q AE")
		
		plt.subplot(3,4,5)
		plt.imshow(test_tilde[ii].reshape(npix,npix), interpolation='None', vmin=vmin, vmax=vmax)
		plt.title('Reconstructed')
		
		plt.subplot(3,4,6)
		plt.imshow(residues.reshape(npix,npix), interpolation='None')
		plt.title(r'$\widetilde{N(x)} - N(x)$')
		
		plt.subplot(3,4,7)
		plt.imshow((test_tilde[ii] - test_nonoise[ii]).reshape(npix,npix), interpolation='None')#, vmin=vmin, vmax=vmax)
		plt.title(r'$\widetilde{N(x)} - x$')
		
		plt.subplot(3,4,8)
		plt.imshow((test_tilde[ii] - test_pca_tilde[ii]).reshape(npix,npix), interpolation='None')
		plt.title(r'$\widetilde{N(x)} - PCA({N(x)})$')
		
		plt.subplot(3,4,9)
		plt.imshow(test_pca_tilde[ii].reshape(npix,npix), interpolation='None', vmin=vmin, vmax=vmax)
		plt.title('PCA Reconstructed')
			
		plt.subplot(3,4,10)
		plt.title(r'$\Delta$')
		plt.imshow((test_pca_tilde[ii] - test_dataset[ii]).reshape(npix,npix), interpolation='None')
		
		plt.subplot(3,4,11)
		plt.imshow((test_pca_tilde[ii] - test_nonoise[ii]).reshape(npix,npix), interpolation='None')#, vmin=vmin, vmax=vmax)
		plt.title(r'$PCA({N(x)}) - x$')
		
		
		plt.subplot(3,4,12)
		plt.imshow((test_tilde[ii] / test_pca_tilde[ii]).reshape(npix,npix), interpolation='None')
		plt.title(r'$\widetilde{N(x)} / PCA({N(x)})$')

	plt.show()
	
def analysis(datasets, parameters=None, names=None, outdir='.', do_meas_params=True, save=True):
	import pylab as plt
	import pylae.figures as f
	
	if names is None:
		names = ["signal", "dataset", "reconstr"]
	meas_params = {}
	
	if parameters is None:
		parameters = ['e1', 'e2', 'r2']
	
	fname_out_meas_params = os.path.join(outdir, 'meas_params.pkl')
	
	if do_meas_params:
		for dataset, name in zip(datasets, names):
			npix = int(np.sqrt(np.shape(dataset)[1]))
			print 'Measuring %s...' % name, 
			rstamps = dataset.reshape(np.shape(dataset)[0], npix, npix)
			measurements = measure_stamps(rstamps)
			
			meas_params[name] = measurements
			print 'done.'
			
		utils.writepickle(meas_params, fname_out_meas_params)
	else:
		meas_params = utils.readpickle(fname_out_meas_params)
		
	e_sig = np.sqrt(meas_params["signal"][:,0]**2 + meas_params["signal"][:,1]**2) / 2.
		
	es = np.sqrt(meas_params["reconstr"][:,0]**2 + meas_params["reconstr"][:,1]**2) / 2.
	rmsd_e = utils.rmsd(es, e_sig)
	rmsd_r = utils.rmsd(meas_params["reconstr"][:,2], meas_params["signal"][:,2])/np.nanmean(meas_params["signal"][:,2])
	sem_e = np.std(es - e_sig)
	sem_r = np.std((meas_params["reconstr"][:,2] - meas_params["signal"][:,2])/np.nanmean(meas_params["signal"][:,2]))
	
	res = [rmsd_e, rmsd_r, sem_e, sem_r]# {'rmsd_e': rmsd_e, 'rmsd_r': rmsd_r, 'sem_e': sem_e, 'sem_r': sem_r}
	
	for idd, npa in enumerate(parameters):
		x = meas_params["signal"][:,idd]
		p = meas_params["reconstr"][:,idd]
		y = p - meas_params["signal"][:,idd]
		
		(m, c) = scipy.polyfit(x, y, 1)
		mean = np.nanmean(y)
		std = np.nanstd(p)
		#res = {'%sm' % npa: m, '%sc' % npa: c, '%smean' % npa: mean, '%sstd' % npa: std}
		res.append(m)
		res.append(c)
		res.append(mean)
		res.append(std)
	
	fig = plt.figure(figsize=(16,12))
	
	plt.subplot(241)
	plt.scatter(meas_params["signal"][:,0], meas_params["reconstr"][:,0]-meas_params["signal"][:,0],)
	plt.subplot(242)
	plt.scatter(meas_params["signal"][:,1], meas_params["reconstr"][:,1]-meas_params["signal"][:,1],)
	plt.subplot(243)
	plt.scatter(meas_params["signal"][:,2], meas_params["reconstr"][:,2]-meas_params["signal"][:,2],)
	plt.subplot(244)
	plt.scatter(meas_params["signal"][:,3], meas_params["reconstr"][:,3])

	plt.subplot(245)
	plt.scatter(meas_params["dataset"][:,0], meas_params["reconstr"][:,0]-meas_params["dataset"][:,0],)
	plt.subplot(246)
	plt.scatter(meas_params["dataset"][:,1], meas_params["reconstr"][:,1]-meas_params["dataset"][:,1],)
	plt.subplot(247)
	plt.scatter(meas_params["dataset"][:,2], meas_params["reconstr"][:,2]-meas_params["dataset"][:,2],)
	plt.subplot(248)
	plt.scatter(meas_params["dataset"][:,3], meas_params["reconstr"][:,3])
	
	if save:
		f.savefig(os.path.join(outdir, 'results'), fig)

	return res
	