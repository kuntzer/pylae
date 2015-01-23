''' 
AIM:	Provide several specific functions to save beautiful figures
REMARKS: in general fancy means latex interpreter (font is serif, Palatino) and generates *.eps and *.pdf
'''
###################################################################################################

def savefig(fname,fig,fancy=False, dpi=300, transparent=True):
	import os
	import subprocess

	fig.savefig(fname+'.png',dpi=dpi)

	if fancy: 
		fig.savefig(fname+'.eps',dpi=dpi,transparent=transparent)
		os.system("epstopdf "+fname+".eps")
		command = 'pdfcrop %s.pdf' % fname
		subprocess.check_output(command, shell=True)
		os.system('mv '+fname+'-crop.pdf '+fname+'.pdf')
	

def set_fancy(font='Palatino'):
	"""
	:param font: which latex font to use. Have a look here http://www.tug.dk/FontCatalogue/
	"""
	from matplotlib import rc
	rc('font',**{'family':'serif','serif':[font],'size':16})
	rc('text', usetex=True)