PYLAE: pythonic library for auto-encoder 
========================================

This code provides a fully Python implementation of Auto-Encoders using different back-propagation algorithms and different cost functions.

The library is intended to be omni-purpose, but is developed for astrophysical applications.

Warning: this is undocumented **work in progress**! You're welcome to contact me if interested or if you have any comments, but don't expect anything useable in here for now.

Current version --- 30 May 2016
-------------------------------

__previous version:__ 20160511

__Main changes:__
* (feature/20160603) Added sparsity constraint
* (feature/20160531) Included dropout, but perfromance seems to be very low
* (feature/20160530) dA configured in L2 error now handles ReLU activation functions, gd and cd1 should be fine too, dautoencoder and autoencoder are changed as well.
* (fix) Adding gaussian noise now normalised.

__Notes:__
* (note/20160531) The normalisation is very important in the case of the AE, not so much in the case of the PCA. If the normalisation factors change by a small fraction, results may degrade very fast.

__Known issues:__
* Multilayered AEs seem to perform badly on PSFs, but nicely on MNIST. Is that because the PSFs are too simple? For now this seems the best explanation. 
* Are dropouts working as they should?
