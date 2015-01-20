"""
A simple galsim script to generate artificial PSFs.
"""

###################################################################################################
# Imports
import numpy as np
import galsim
import pylab as plt
import os

outdir = 'data'
run_name = 'smalldev-noisy'

# Number of PSFs
n = 3000

# Pixel scale in arcsec / pixel
pixel_scale = 0.1

# PSF parameters
psf_re = 0.5 # arcsec
psf_beta = 5 

# n x n pixels
image_size = 32 

# Parameter preview ?
parampre = False

# Noise level
noise = 0.0005

###################################################################################################
# Initialization


def rnd(ud):
	
	return (ud() - 0.5) * 0.2;

if parampre :
	r = []
	for i in range(10000):
		ud = galsim.UniformDeviate() 
		rr = psf_re*((ud() - 0.5) * 0.3 + 1.)
		r.append(rr)
	
	r = np.asarray(r)
	print np.amin(r), np.amax(r)
	
	plt.figure()
	plt.hist(r)
	plt.show()	
	exit()

psffname = "%s/psfs-%s.dat" % (outdir, run_name)
truthfname = "%s/truth-%s.dat" % (outdir, run_name)

if (os.path.exists(psffname) or os.path.exists(truthfname))  and False :
	print 'Either of the following files exists :'
	print psffname
	print truthfname
	raise IOError("Files already exist")

output = np.zeros([n, image_size*image_size])
truth = np.zeros([n, 3])

###################################################################################################
# Core script
psf_re_i = psf_re
for i in range(n):
	ud = galsim.UniformDeviate() 
	g1 = rnd(ud)
	g2 = rnd(ud)
	psf_re = psf_re_i*((ud() - 0.5) * 0.3 + 1.)

	print '%5d : g1=%+1.5f, g2=%+1.5f, psf_re=%1.5f' % (i, g1, g2, psf_re)

	psf = galsim.Moffat(beta=psf_beta, flux=1., half_light_radius=psf_re)
	psf = psf.shear(g1=g1, g2=g2)
	
	image = galsim.ImageF(image_size, image_size)
	
	psf.drawImage(image=image, scale=pixel_scale)
	image.addNoise(galsim.GaussianNoise(sigma=noise))
	output[i] = image.array.flatten()
	truth[i] = [g1, g2, psf_re]

	if i in [0,1]:
		plt.figure()
		plt.imshow(image.array, interpolation="nearest")
		plt.show()
		
###################################################################################################
# Write to disk
np.savetxt(psffname, output, delimiter=',')
np.savetxt(truthfname, truth, delimiter=',')
