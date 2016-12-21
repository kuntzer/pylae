import pylab as plt

import pylae


# Loading the data and pre-processing
N_train = 500
N_test = 100

images = pylae.utils_mnist.load_MNIST_images('mnist-data/train-images.idx3-ubyte')
images = images.T
ids_train = range(N_train)
ids_test = range(N_train, N_train+N_test)

images_train = images[ids_train]
images_test = images[ids_test]

# Preparing the SAE
dA = pylae.dA.AutoEncoder("sae_mnist")

architecture = [128, 64, 8]
layers_type = ["SIGMOID", "SIGMOID", "SIGMOID", "SIGMOID"]
cost_fct = 'cross-entropy'

# Define what training we should do
do_pre_train = False
do_train = False
iters = 20

if do_pre_train:
	dA.pre_train(images_train, architecture, layers_type, iterations=iters, mini_batch=0, corruption=None)
	pylae.utils.writepickle(dA.rbms, "%s/dA/layers.pkl" % dA.name)
else:
	rbms = pylae.utils.readpickle("%s/dA/layers.pkl" % dA.name)
	dA.set_autoencoder(rbms)
	dA.is_pretrained = True

if do_train:
	dA.fine_tune(images_train, iterations=1000, regularisation=iters, sparsity=0.0, beta=0., corruption=None, cost_fct=cost_fct)
	dA.save()
else:
	dA = pylae.utils.readpickle("%s/dA/ae.pkl" % dA.name)
	print 'AE loaded!'

dA.display_train_history()
dA.display_network(0)