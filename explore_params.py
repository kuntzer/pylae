import pylae
import pylae.utils as utils

import numpy as np
import multiprocessing
import pylab as plt
###################################################################################################

def train(train_data, params, layers_type):
	architecture = params
	print "Starting", architecture
	ae = pylae.autoencoder.AutoEncoder()
	ae.pre_train(train_data, architecture, layers_type, learn_rate={'SIGMOID':0.0034, 'LINEAR':0.0034/10.}, 
				initialmomentum=0.53,finalmomentum=0.93, iterations=2000, mini_batch=100, regularisation=0.001)
	ae.backpropagation(train_data, iterations=1000, learn_rate=0.13, momentum_rate=0.83)
	
	return ae

def cost(ae, test_data):

	test_ell = []
	recon_test_ell = []
	
	reconstructest = ae.feedforward(test_data)

	size = np.sqrt(np.shape(test_data)[1])

	for ii in range(np.shape(test_data)[0]):
		
		img = test_data[ii].reshape(size,size)
		try:
			g1, g2 = utils.get_ell(img)
		except:
			continue
		test_ell.append([g1, g2, np.sqrt(g1*g1 + g2*g2)])
		
		img = reconstructest[ii].reshape(size,size)
		try:
			g1, g2 = utils.get_ell(img)
		except: 
			continue
		recon_test_ell.append([g1, g2, np.sqrt(g1*g1 + g2*g2)])
	
	test_ell = np.asarray(test_ell)
	recon_test_ell = np.asarray(recon_test_ell)

	ae_ell_error = recon_test_ell[:,2] - test_ell[:,2]
	
	ae_error = np.sqrt(np.mean(ae_ell_error * ae_ell_error))

	return ae_error

def worker(params):
	architecture = params
	layers_type = ["SIGMOID"] * len(architecture) + ["LINEAR"]
	
	ae = train(train_data, params, layers_type)

	try:
		c = cost(ae, test_data)
	except IndexError :
		c = -0.0001
	return c

###################################################################################################

dataset = np.loadtxt("data/psfs-smalldev.dat", delimiter=",")
dataset, low, high = utils.normalise(dataset)

datasize = np.shape(dataset)[0]
datasize = 2000
trainper = 0.7
ind = np.int(trainper * datasize)

train_data = dataset[0:ind]
test_data = dataset[ind:datasize]

nb_params = 32

explore = True

min_nb_neuron = 15
#architectures=[[np.int(a), min_nb_neuron] for a in np.linspace(2*min_nb_neuron, 500, nb_params)]

#architecture = [512, 16]
#params = np.logspace(-6,-0.09, nb_params)
#params = np.linspace(0.001,1., nb_params)

architectures = []
nnn = np.linspace(min_nb_neuron, 784, nb_params)
nl = [1,2,3]
for nn in nnn:
	for l in nl:
		maxn = nn
		minn = min_nb_neuron
		
		aa = np.linspace(minn, maxn, l+1)
		archi = []#[np.int(max)]
		for a in aa[::-1]:
			archi += [np.int(a)]
		architectures.append(archi)
params = architectures

if explore:
	ncpu=multiprocessing.cpu_count()
	pool = multiprocessing.Pool(processes=ncpu)
	
	res = pool.map(worker, params)
	res = np.asarray(res)
	nn = []
	score = []
	
	for ii in range(len(params)):
		nn.append(ii)
		score.append(res[ii])
	
	utils.writepickle([nn, score], "diagnostic_explore_params.pkl")
	
	pool.close()
	pool.join()
else:
	nn, score = utils.readpickle([nn, score], "diagnostic_explore_params.pkl")

plt.figure()
plt.plot(nn, score, lw=2)
plt.grid()
plt.show()