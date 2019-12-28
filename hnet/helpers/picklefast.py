""" This function loads and saves a pickle file.

	A= picklefast.load(filename, <optional>)
	A= picklefast.save(filename, <optional>)

 INPUT:
   filename:       String: Path to pickle file

 OPTIONAL

   verbose:        Boolean [0,1] or [True,False]
                   False: No (default)
                   True: Yes

 OUTPUT
	output

 DESCRIPTION
   Load or save a pickle file

 EXAMPLE
   import GENERAL.picklefast as picklefast
   importlib.reload(picklefast)

   filename='c:/temp/tes1t.pkl'
   var=[1,2,3,4,5]
   A    = picklefast.save(filename,var)
   var1 = picklefast.load(filename)

 SEE ALSO
   pickle
"""

#--------------------------------------------------------------------------
# Name        : pickle.py
# Version     : 1.0
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Date        : Aug. 2018
#--------------------------------------------------------------------------

import pickle
import os

#%% Save pickle file
def save(filename, var, verbose=True):
    # Make empty pickle file
    outfile = open(filename,'wb')
    # Write and close
    pickle.dump(var,outfile)
    outfile.close()
    if os.path.isfile(filename):
        if verbose: print('[PICKLE] Pickle file saved: [%s]' %filename)
        out=True
    else:
        if verbose: print('[PICKLE] Pickle file could not be saved: [%s]' %filename)
    return(out)
    
#%% Load pickle file
def load(filename, verbose=False):
    out=None
    if os.path.isfile(filename):
        if verbose: print('[PICKLE] Pickle file loaded: [%s]' %filename)
        pickle_off = open(filename,"rb")
        out = pickle.load(pickle_off)
    else:
        if verbose: print('[PICKLE] Pickle file does not exists: [%s]' %filename)    
    return(out)
    
