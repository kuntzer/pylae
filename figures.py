''' 
AIM:	Provide several specific functions to save beautiful figures
'''

import os 
import subprocess
from matplotlib import rc

###################################################################################################

def savefig(fname,fig,fancy=False, dpi=300, transparent=True):

	fig.savefig(fname+'.png',dpi=dpi)

	if fancy: 
		fig.savefig(fname+'.eps',dpi=dpi,transparent=transparent)
		os.system("epstopdf "+fname+".eps")
		command = 'pdfcrop %s.pdf' % fname
		subprocess.check_output(command, shell=True)
		os.system('mv '+fname+'-crop.pdf '+fname+'.pdf')

	
def set_fancy(font='Computer Modern', txtsize=16):
	"""
	:param font: which latex font to use. Have a look here http://www.tug.dk/FontCatalogue/
	:param txtsize: the text size
	"""
	
	rc('font',**{'size':txtsize})
	rc('font', **{'family': 'serif', 'serif': [font]})
	rc('text', usetex=True)
