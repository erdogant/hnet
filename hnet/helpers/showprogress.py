"""Show progress bar

FUNCTION showprogress
	A = showprogress(i, max_i, <optional>)

 INPUT:
   i:          i-th index
 
   max_i:      Maximum nr of iterations to do in the e.g., for-loop


 OPTIONAL

   steps :     Integer (default: 25)
               Define how many characters needs to printed to show the progress
 
   barwidth :  Integer (default: 25)
               Define the width of the bar

   printchar   Char (default: '=')
               Define the char that is printend
 
   tictoc:	  toc
               Include timestamp
 
   newline:	  Boolean Make new line after showing the estimated time
               [1] Yes (default)
               [0] No

 OUTPUT
	output

 DESCRIPTION
   Progresbar

 EXAMPLE
   from showprogress import showprogress

   maxdata=100
   for i in range(0,maxdata):
       tic()
       time.sleep(0.2)
       showprogress(i,maxdata,steps=25, barwidth=25, printchar='>', tictoc=toc())

 SEE ALSO
   tic, toc, tqdm

--------------------------------------------------------------------------
 Name        : showprogress.m
 Version     : 1.0
 Author      : E.Taskesen
 Contact     : erdogant@gmail.com
 Date        : Sep. 2017
--------------------------------------------------------------------------
"""

#%% Libraries
# import sys, os, importlib
import numpy as np
import sys
import datetime as datetime

#%%
def showprogress(i, maxdata, steps=25, barwidth=25, printchar='=', newline=False, tictoc=0):
    #%% Make dictionary to store Parameters
    Param = {}
    Param['steps']    = steps
    Param['barwidth'] = barwidth
    Param['char']     = printchar
    Param['tictoc']   = tictoc
    Param['newline']  = newline

    del steps, barwidth, printchar, tictoc
        
	#%% Correct index i starts at 0
    endwith=''
    if Param['newline']: endwith='\n'
    maxdata = max(maxdata,1)
    i=i+1

    #%% Max data bar
    if i==maxdata:
        # Erase line
        sys.stdout.write('\r')
        getPerc=1
       # Determine number of '='
        getSteps = int(Param['barwidth']*getPerc)
        getRest  = Param['barwidth'] - getSteps
        # Make bar 
        StartBar   = ("[%s" %(Param['char']*getSteps ))
        StopBar    = ("%s=]" %(' '*getRest))
        ProgStatus = str(np.round(getPerc,2)*100)
        ProgStatus = ProgStatus[0:3]+"%"
        # writebar
        TotBar = StartBar + StopBar + " " + ProgStatus
        sys.stdout.write(TotBar)
        #return()
     #end
     
     #%% Only print if within the steps:
    if np.mod(i, int(np.round(maxdata/Param['steps'])))==0:
        # Erase line
        sys.stdout.write('\r')
        
        # Compute overal progress
#        if i==maxdata:
#            getPerc = 100
#        else:
        if maxdata!=0:
            getPerc = (i/maxdata)
        #end

        # Determine number of '='
        getSteps = int(Param['barwidth']*getPerc)
        getRest  = Param['barwidth'] - getSteps
        
        # Make bar 
        StartBar   = ("[%s" %(Param['char']*getSteps ))
        StopBar    = ("%s]" %(' '*getRest))
        ProgStatus = str(np.round(getPerc,2)*100)
        ProgStatus = ProgStatus[0:3]+"%"

        # Add ETA
        ETA=" "
        if Param['tictoc']>0:
            # Determine nr seconds todo
            todoSec = (Param['tictoc'])*(maxdata-i)
            # Compute ETA
            ETA = datetime.datetime.now() + datetime.timedelta(seconds=todoSec)
            ETA = " | " + ETA.strftime("%d-%m-%Y %H:%M:%S")
        #end
    #end
        
        #%% Final bar
        TotBar = StartBar + StopBar + " " + ProgStatus + ETA + endwith
        sys.stdout.write(TotBar)
        sys.stdout.flush()

       
	#%% END
