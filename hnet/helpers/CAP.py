""" This function computes Cumulitive Accuracy Profile (CAP) to measure the performance in a two class classifier

	A= CAP.plot(data, <optional>)

 INPUT:
   data:           datamatrix
                   rows    = features
                   colums  = samples
 OPTIONAL

   verbose:        Integer [0..5] if verbose >= DEBUG: print('debug message')
                   0: (default)
                   1: ERROR
                   2: WARN
                   3: INFO
                   4: DEBUG
                   

 OUTPUT
	output

 DESCRIPTION
  The CAP Curve analyse how to effectively identify all data points of a given class using minimum number of tries.
  This function computes Cumulitive Accuracy Profile (CAP) to measure the performance of a classifier.
  It ranks the predicted class probabilities (high to low), together with the true values.
  With that, it computes the cumsum which is the final line.
  

 EXAMPLE
   %reset -f
   %matplotlib auto
   import pandas as pd
   from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
   from TRANSFORMERS.df2onehot import df2onehot
   import SUPERVISED.twoClassSummary as twoClassSummary
   from sklearn.model_selection import train_test_split
   gb=GradientBoostingClassifier()

   import VIZ.CAP as CAP
   
   ######## Load some data ######
   df=pd.read_csv('../DATA/OTHER/titanic/titanic_train.csv')
   dfc=df2onehot(df)[0]
   dfc.dropna(inplace=True)
   y=dfc['Survived'].astype(float).values
   del dfc['Survived']
   [X_train, X_test, y_train, y_test]=train_test_split(dfc, y, test_size=0.2)

   # Prediction
   model=gb.fit(X_train, y_train)
   P=model.predict_proba(X_test)
   twoClassSummary.allresults(y_test, P[:,1])

   A = CAP.plot(y_test, P[:,1])

 SEE ALSO
   ROCplot, twoClassSummary
"""

#--------------------------------------------------------------------------
# Name        : CAP.py
# Version     : 1.0
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Date        : April. 2019
#--------------------------------------------------------------------------

#%% Libraries
import numpy as np
import matplotlib.pyplot as plt

#%% CAP
def plot(y_true, y_pred, label='Classifier', ax=None, showfig=True):
    config = dict()
    config['label']=label
    fontsize=14
    
    total = len(y_true)
    class_1_count = np.sum(y_true)
    #class_0_count = total - class_1_count
    
    if showfig:
        if isinstance(ax, type(None)):
            fig,ax=plt.subplots(figsize = (20, 12))
        
        ax.plot([0, total], [0, class_1_count], c = 'navy', linestyle = '--', label = 'Random Model')
        ax.set_xlabel('Total observations', fontsize = fontsize)
        ax.set_ylabel('Class observations', fontsize = fontsize)
        ax.set_title('Cumulitive Accuracy Profile (CAP)', fontsize = fontsize)
        ax.grid(True)
    
        # A perfect model is one which will detect all class 1.0 data points in the same number of tries as there are class 1.0 data points. 
        # It takes exactly 58 tries for the perfect model to identify 58 class 1.0 data points.
        ax.plot([0, class_1_count, total], [0, class_1_count, class_1_count], c='grey', linewidth=1, label = 'Perfect Model')

    # Probs and y_test are zipped together. 
    # Sort this zip in the reverse order of probabilities such that the maximum probability comes first and then lower probabilities follow. 
    # I extract only the y_test values in an array and store it in model_y.
    prob_and_true=sorted(zip(y_pred, y_true), reverse = True)
    model_y = [y for _, y in prob_and_true]
    
    # creates an array of values while cumulatively adding all previous values in the array to the present value. 
    y_values = np.append([0], np.cumsum(model_y))
    x_values = np.arange(0, total + 1)
    
    # Plot accuracy
    if showfig:
        ax.plot(x_values, y_values, c='darkorange', label=config['label'], linewidth=2)
        ax.legend(loc = 'lower right', fontsize = fontsize)

    return(max(y_values))

