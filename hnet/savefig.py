""" This function saves figures in PNG format."""
#--------------------------------------------------------------------------
# Name        : savefig.py
# Version     : 1.0
# Author      : E.Taskesen
# Date        : Sep. 2017
#--------------------------------------------------------------------------
# Libraries
from os import mkdir
from os import path

#%%
def savefig(fig, filepath, dpi=100, transp=False, verbose=0):
    """Saving figure.

    Parameters
    ----------
    fig : Figure object
        Figure to be saved.
    filepath : str
        path to store file.
    dpi : TYPint, The default is 100.
        Resolution fo the figure to storein Dotch Per Inch. 
    transp : bool, The default is False.
        Set background transparancy.
    verbose : int [1-5], default: 3
        Print information to screen. 0: nothing, 1: Error, 2: Warning, 3: information, 4: debug, 5: trace.

    Returns
    -------
    bool: status of succesfull saving.

    """
    success=False # Returns 1 if succesful
    # Write figure to path
    if filepath!="":
        # Check dir
        [getpath, getfilename] = path.split(Param['filepath'])
        if path.exists(getpath)==False:
            mkdir(getpath)
        
        # save file
        fig.savefig(filepath, dpi=dpi, transparent=transp, bbox_inches='tight')
        success=True
        
    return(success)
