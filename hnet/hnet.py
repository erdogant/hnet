"""HNET: Hypergeometric-networks. This function computes significance of the response variable with catagorical/numerical variables.

  import hnet as hnet

  out              = hnet.fit(df, <optional>)
  outy             = hnet.enrichment(df, y, <optional>)
  G                = hnet.plot_heatmap(out, <optional>)
  G                = hnet.plot_network(out, <optional>)
  G                = hnet.plot_d3graph(out, <optional>)
  rules            = hnet.combined_rules(out)
  [scores, adjmat] = hnet.compare_networks(out1['adjmat'], out['adjmat'])
  adjmatSymmetric  = hnet.to_symmetric(out)


 INPUT:
   df:             [pd.DataFrame] Pandas DataFrame for which every row contains samples with featureson the columns

                      f1  ,f2  ,f3
                   s1 0   ,0   ,1
                   s2 0   ,1   ,0
                   s3 1   ,1   ,0

    y              [numpy array] Vector of labels
                   [0,1,0,1,1,2,1,2,2,2,2,0,0,1,0,1,..]
                   ['aap','aap','boom','mies','boom','aap',..]


 OPTIONAL

   y_min:          [Integer] [samples>=y_min] Minimal number of samples in a group. All groups with less then y_min samples are labeled as _other_ and are not used in the model.
                   10  (default)
                   None

   alpha:          Float [0..1] Alpha that serves as cuttoff point for significance
                   0.05 : (default)
                   1    : (for all results)

   dtypes:         [list strings] in the form ['cat','num',''] of length y. By default the dtype is determined based on the pandas dataframe. Empty ones [''] are skipped.
                   ['cat','cat','num','','cat',...]

   perc_min_num: [float] Force column (int or float) to be numerical if unique non-zero values are above percentage.
                   None
                   0.8 (default)

   k:              [integer][1..n] Number of combinatoric elements to create for the n features
                   1 (default)

   target:         Target value for response variable y should have dtype y.
                   None : (default) All values for y are tested
                   1    :  In a two class model

   dropna:         [Bool] [True,False] Drop rows/columns in adjacency matrix that showed no significance
                   True  (default)
                   False

   specificity:    [string]: To configure how numerical data labels are stored. Setting this variable can be of use in the 'structure_learning' function for the creation of a network ([None] will glue most numerical labels together whereas [high] mostly will not).
                   None    : No additional information in the labels
                   'low'   : 'high' or 'low' are included that represents significantly higher or lower assocations compared to the rest-group.
                   'medium': (default) 'high' or 'low' are included with 1 decimal behind the comma.
                   'high'  : 'high' or 'low' are included with 3 decimal behind the comma.

   multtest:       [String]:
                    None            : No multiple Test
                   'bonferroni'     : one-step correction
                   'sidak'          : one-step correction
                   'holm-sidak'     : step down method using Sidak adjustments
                   'holm'           : step-down method using Bonferroni adjustments (default)
                   'simes-hochberg' : step-up method  (independent)
                   'hommel'         : closed method based on Simes tests (non-negative)
                   'fdr_bh'         : Benjamini/Hochberg  (non-negative)
                   'fdr_by'         : Benjamini/Yekutieli (negative)
                   'fdr_tsbh'       : two stage fdr correction (non-negative)
                   'fdr_tsbky'      : two stage fdr correction (non-negative)

   excl_background:[String]: String name to exclude from the background
                   None     (default)
                   ['0.0']: To remove catagorical values with label 0

   savepath:       [string]: Direcotry or Full path of figure to save to disk. If only dir is given, filename is created.
                   None    : (default) do not save
                   './hnet/figs/'
                   './hnet/figs/hnet_fig.png'

   verbose:        [Integer] [0..5] if verbose >= DEBUG: print('debug message')
                   0: (default)
                   1: ERROR
                   2: WARN
                   3: INFO
                   4: DEBUG

 Requirements
   See Requirements.txt


 Output
 ------
   model

 Descriptions
 -----------
   Automatically generates networks from datasets with mixed datatypes return
   by an unknown function. These datasets can range from generic dataframes to
   nested data structures with lists, missing values and enumerations.
   I solved this problem to minimize the amount of configurations required
   while still gaining many benefits of having schemas available.

   The response variable (y) should be a vector with the same number of samples
   as for the input data. For each column in the dataframe significance is
   assessed for the labels in a two-class approach (y=1 vs y!=1).
   Significane is assessed one tailed; only the fit for y=1 with an
   overrepresentation. Hypergeometric test is used for catagorical values
   Wilcoxen rank-sum test for numerical values

 Example
 -------
   See test.py


"""

# --------------------------------------------------------------------------
# Name        : hnet.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Date        : Dec. 2019
# --------------------------------------------------------------------------

# %% Libraries
# Basics
import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import itertools
# Known libraries
from scipy.stats import hypergeom, ranksums
import statsmodels.stats.multitest as multitest
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import networkx as nx
label_encoder = LabelEncoder()
import matplotlib.pyplot as plt
# Custom package
from d3graph import d3graph
from ismember import ismember
import imagesc as imagesc
# Custom helpers
from hnet.helpers.set_dtypes import set_dtypes, set_y
from hnet.helpers.df2onehot import df2onehot
# VIZ
from hnet.helpers.savefig import savefig
import hnet.helpers.network as network
# Warnings
import warnings
warnings.filterwarnings("ignore")

#%% Structure learning across all variables
def fit(df, alpha=0.05, y_min=10, k=1, multtest='holm', dtypes='pandas', specificity='medium', perc_min_num=0.8, dropna=True, excl_background=None, verbose=3):
    assert isinstance(df, pd.DataFrame), 'Input data [df] must be of type pd.DataFrame()'
    param=dict()
    param['k']=k
    param['alpha']=alpha
    param['y_min']=y_min
    param['multtest']=multtest
    param['fillna']=True
    param['specificity']=specificity
    param['perc_min_num']=perc_min_num
    param['dropna']=dropna
    param['excl_background']=excl_background
    param['verbose']=verbose
    
    # newline=True if verbose>=4 else False
    df.columns = df.columns.astype(str)
    # Remove columns without dtype
    [df, dtypes]=remove_columns_without_dtype(df, dtypes, verbose=param['verbose'])
    # Make onehot matrix for response variable y
    out_onehot=df2onehot(df, dtypes=dtypes, y_min=param['y_min'], hot_only=True, perc_min_num=param['perc_min_num'], excl_background=param['excl_background'], verbose=param['verbose'])
    dtypes=out_onehot['dtypes']
    # Some check before proceeding
    assert (not out_onehot['onehot'].empty) or (not np.all(np.isin(dtypes,'num'))), '[HNET] ALL data is excluded from the dataframe! There should be at least 1 categorical value!'
    assert df.shape[1]==len(dtypes), '[HNET] DataFrame Shape and dtypes length does not match'
    # Make all integer
    out_onehot['onehot']=out_onehot['onehot'].astype(int)
    # Add combinations
    [X_comb, X_labx, X_labo]=make_n_combinations(out_onehot['onehot'], out_onehot['labx'], param['k'], param['y_min'], verbose=param['verbose'])
    # Print some
    if param['verbose']>=3: print('[HNET] Structure learning across [%d] features.' %(X_comb.shape[1]))
    # Get numerical columns
    colNum      = df.columns[df.dtypes=='float64'].values
    simmat_labx = np.append(X_labo, colNum).astype(str)
    simmat_padj = pd.DataFrame(index=np.append(X_comb.columns, colNum).astype(str), columns=np.append(X_comb.columns, colNum).astype(str)).astype(float)


    # Here we go! in paralel!
    # from multiprocessing import Pool
    # nr_succes_pop_n=[]
    # with Pool(processes=os.cpu_count()-1) as pool:
    #     for i in range(0,X_comb.shape[1]):
    #         result = pool.apply_async(do_the_math, (df, X_comb, dtypes, X_labx, param, i,))
    #         nr_succes_pop_n.append(result)
        
    #     results = [result.get() for result in result_objs]
    #     print(len(results))
        


    # Here we go! Over all columns now
    count=0
    nr_succes_pop_n=[]
    for i in tqdm(range(0,X_comb.shape[1]), disable=(True if param['verbose']==0 else False)):
        [nr_succes_i, simmat_padj, simmat_labx] = do_the_math(df, X_comb, dtypes, X_labx, simmat_padj, simmat_labx, param, i)
        nr_succes_pop_n.append(nr_succes_i)

    # Message
    if param['verbose']>=3: print('[HNET] Total number of computations: [%.0d]' %(count))

    # Post processing
    [simmat_padj, nr_succes_pop_n, adjmatLog, simmat_labx] = post_processing(simmat_padj, nr_succes_pop_n, simmat_labx, param)

    # Store
    out=dict()
    out['simmatP']    = simmat_padj
    out['simmatLogP'] = adjmatLog
    out['labx']       = simmat_labx.astype(str)
    out['dtypes']     = np.array(list(zip(df.columns.values.astype(str), dtypes)))
    out['counts']     = nr_succes_pop_n
    out['rules']      = combined_rules(out, verbose=0)

    # Return
    return(out)

#%% Compute fit
def enrichment(df, y, y_min=None, alpha=0.05, multtest='holm', dtypes='pandas', specificity='medium', verbose=3):
    assert isinstance(df, pd.DataFrame), 'Data must be of type pd.DataFrame()'
    assert len(y)==df.shape[0], 'Length of [df] and [y] must be equal'
    assert 'numpy' in str(type(y)), 'y must be of type numpy array'
    
	# DECLARATIONS
    config = dict()
    config['verbose']  = verbose
    config['alpha']    = alpha
    config['multtest'] = multtest
    config['specificity'] = specificity

    if config['verbose']>=3: print('[HNET] Start making fit..')
    df.columns = df.columns.astype(str)
    # Set y as string
    y = set_y(y, y_min=y_min, verbose=config['verbose'])
    # Determine dtypes for columns
    [df, dtypes] = set_dtypes(df, dtypes, verbose=config['verbose'])
    # Compute fit
    out = compute_significance(df, y, dtypes, specificity=config['specificity'], verbose=config['verbose'])
    # Multiple test correction
    out = multipletestcorrection(out, config['multtest'], verbose=config['verbose'])
    # Keep only significant ones
    out = filter_significance(out, config['alpha'], multtest)
    # Make dataframe
    out = pd.DataFrame(out)
    # Return
    if config['verbose']>=3: print('[HNET] Fin')
    return(out)

#%% Compute significance
def compute_significance(df, y, dtypes, specificity=None, verbose=3):
    out=[]
    # Run over all columns
    for i in range(0, df.shape[1]):
        if (i>0) and (verbose>=3): print('')
        if verbose>=3: print('[HNET] Analyzing [%s] %s' %(dtypes[i], df.columns[i]), end='')
        colname = df.columns[i]
        
        # Clean nan fields
        [datac, yc] = nancleaning(df[colname], y)
        # In a two class model, remove 0-catagory
        uiy = np.unique(yc)
        # No need to compute _other_ because it is a mixed group that is auto generated based on y_min
        uiy=uiy[uiy!='_other_']

        if len(uiy)==1 and (uiy=='0'):
            if verbose>=4: print('[HNET] The response variable [y] has only one catagory; [0] which is seen as the negative class and thus ignored.')
            uiy=uiy[uiy!='0']
        
        if len(uiy)==2:
            if verbose>=4: print('[HNET] The response variable [y] has two catagories, the catagory 0 is seen as the negative class and thus ignored.')
            uiy=uiy[uiy!='0']
            uiy=uiy[uiy!='False']
            uiy=uiy[uiy!='false']
            
        # Run over all target values
        for j in range(0, len(uiy)):
            target = uiy[j]
            # Catagorical
            if dtypes[i]=='cat':
                datacOnehot=pd.get_dummies(datac)
                # Remove background column if 2 columns of which 0.0 also there
                if (datacOnehot.shape[1]==2) & (np.any(np.isin(datacOnehot.columns,'0.0'))):
                    datacOnehot.drop(labels=['0.0'], axis=1, inplace=True)
                # Run over all unique entities/cats in column for target vlue
                for k in range(0,datacOnehot.shape[1]):
                    outtest = prob_hypergeo(datacOnehot.iloc[:,k], yc==target)
                    outtest.update({'y':target})
                    outtest.update({'category_name':colname})
                    out.append(outtest)
            
            # Numerical
            if dtypes[i]=='num':
                outtest = prob_ranksums(datac, yc==target, specificity=specificity)
                outtest.update({'y':target})
                outtest.update({'category_name':colname})
                out.append(outtest)
            # Print dots
            if verbose>=3: print('.',end='')
    
    if verbose>=3: print('')
    return(out)

#%% Wilcoxon Ranksum test
def prob_ranksums(datac, yc, specificity=None):
    P=np.nan
    zscore=np.nan
    datac=datac.values
    getsign=''
    
    # Wilcoxon Ranksum test
    if sum(yc==True)>1 and sum(yc==False)>1:
        [zscore,P]=ranksums(datac[yc==True], datac[yc==False])
    
    # Store
    out=dict()
    out['P']=P
    out['logP']=np.log(P)
    out['zscore']=zscore
    out['popsize']=len(yc)
    out['total_target_samples']=np.sum(yc==True)
    out['total_other_samples']=np.sum(yc==False)
    out['dtype']='numerical'

    if np.isnan(zscore)==False and np.sign(zscore)>0:
        getsign='high_'
    else:
        getsign='low_'
    
    if specificity=='low':
        out['category_label']=getsign[:-1]
    elif specificity=='medium':
        out['category_label']=getsign+str(('%.1f' %(np.median(datac[yc==True]))))
    elif specificity=='high':
        out['category_label']=getsign+str(('%.3f' %(np.median(datac[yc==True]))))
    else:
        out['category_label']=''
    
    return(out)

#%% Hypergeometric test
def prob_hypergeo(datac, yc):
    '''
    Suppose you have a lot of 100 floppy disks (M), and you know that 20 of them are defective (n).
    What is the prbability of drawing zero to 2 floppy disks (N=2), if you select 10 at random (N).
    P=hypergeom.sf(2,100,20,10)
    '''
    P=np.nan
    logP=np.nan
    M = len(yc) # Population size: Total number of objects, eg total number of genes; 10000
    n = np.sum(datac) #Number of successes in populatoin, known in pathway, eg 2000
    N = np.sum(yc) # sample size: Random variate, eg clustersize or groupsize, over expressed genes, eg 300
    X = np.sum(np.logical_and(yc, datac))-1 # Let op, de -1 is belangrijk omdatje P<X wilt weten ipv P<=X. Als je P<=X doet dan kan je vele false positives krijgen als bijvoorbeeld X=1 en n=1 oid
    
    # Test
    if np.any(yc) and (X>0):
        P    = hypergeom.sf(X, M, n, N)
        logP = hypergeom.logsf(X, M, n, N)
    
    # Store
    out=dict()
    out['category_label']=datac.name
    out['P']=P
    out['logP']=logP
    out['overlap_X']=X
    out['popsize_M']=M
    out['nr_succes_pop_n']=n
    out['samplesize_N']=N
    out['dtype']='categorical'
    
    return(out)

#%% Make logscale
def logscale(simmat_padj):
    # Set minimum amount
    simmat_padj[simmat_padj==0]=1e-323
    adjmatLog=(-np.log10(simmat_padj)).copy()
    adjmatLog[adjmatLog == -np.inf] = np.nanmax(adjmatLog[adjmatLog != np.inf])
    adjmatLog[adjmatLog == np.inf] = np.nanmax(adjmatLog[adjmatLog != np.inf])
    adjmatLog[adjmatLog == -0] = 0
    return(adjmatLog)

#%% Extract combined rules from structure_learning
def combined_rules(out, verbose=3):
#    assert isinstance(out.get('simmatP',None), type(None)), 'input value should be dictionary containing simmatP.'
    
    from scipy.stats import combine_pvalues
    df_rules = pd.DataFrame(index=np.arange(0,out['simmatP'].shape[0]), columns=['antecedents_labx','antecedents','consequents','Pfisher'])
    df_rules['consequents'] = out['simmatP'].index.values
    
    for i in tqdm(range(0, out['simmatP'].shape[0]), disable=(True if verbose==0 else False)):
        idx=np.where(out['simmatP'].iloc[i,:]<1)[0]
        # Remove self
        idx=np.setdiff1d(idx,i)
        # Store rules
        df_rules['antecedents'].iloc[i] = list(out['simmatP'].iloc[i,idx].index)
        df_rules['antecedents_labx'].iloc[i] = out['labx'][idx]
        # Combine pvalues
        df_rules['Pfisher'].iloc[i] = combine_pvalues(out['simmatP'].iloc[i,idx].values, method='fisher')[1]
        # Showprogress
        # if verbose>=3: showprogress(i, out['simmatP'].shape[0])
    
    # Keep only lines with pvalues
    df_rules.dropna(how='any', subset=['Pfisher'], inplace=True)
    # Sort
    df_rules.sort_values(by=['Pfisher'], ascending=True, inplace=True)
    df_rules.reset_index(inplace=True, drop=True)
    
    return(df_rules)
    
#%% Add columns
def addcolumns(simmat_padj, colnames, Xlabx, catnames):
    I=np.isin(colnames.values.astype(str), simmat_padj.index.values)
    if np.any(I):
        newcols=list((colnames.values[I==False]).astype(str))
        newcats=list((catnames[I==False]).astype(str))

        # Make new columns in dataframe
        for col,cat in zip(newcols, newcats) :
            simmat_padj[col]=np.nan
            Xlabx = np.append(Xlabx, cat)

        
        addrow=pd.DataFrame(index=newcols, columns=simmat_padj.columns.values).astype(float)
        simmat_padj=pd.concat([simmat_padj, addrow])
    return(simmat_padj, Xlabx)

#%% Remove columns without dtype
def remove_columns_without_dtype(df, dtypes, verbose=3):
    if not isinstance(dtypes, str):
        assert df.shape[1]==len(dtypes), 'Columns in df and dtypes should match! [hnet.remove_columns_without_dtype]'
        I=np.isin(dtypes,'')
        if np.any(I):
            remcols=df.columns[I].values
            df.drop(columns=remcols, inplace=True)
            dtypes=list(np.array(dtypes)[(I==False)])
            if verbose>=3: print('[HNET] %.0f columns are removed.' %(len(remcols)))
    
        assert df.shape[1]==len(dtypes), 'Columns in df and dtypes should match! [hnet.remove_columns_without_dtype]'

    return(df, dtypes)

#%% Clean empty rows
def drop_empty(df, Xlabx, verbose=3):
    dfO=df.copy()
    cols=dfO.columns.values
    rows=dfO.index.values
    
    # Set diagonal on nan
    np.fill_diagonal(df.values, np.nan)

    droplabel=[]
    for col in cols:
        if np.any(cols==col):
            if np.all(np.logical_and(df.loc[:,cols==col].isna().values.reshape(-1,1), df.loc[rows==col,:].isna().values.reshape(-1,1))):
                if verbose>=3: print('[HNET] Dropping %s' %(col))
                droplabel.append(col)

    # Remove labels from the original df
    Xlabx=Xlabx[np.isin(dfO.columns, droplabel)==False]
    dfO.drop(labels=droplabel, axis=0, inplace=True)
    dfO.drop(labels=droplabel, axis=1, inplace=True)
    return(dfO, Xlabx)

#%% Do multiple test correction
def multipletestcorrectionAdjmat(adjmat, multtest, verbose=3):
    if verbose>=3: print('[HNET] Multiple test correction using %s' %(multtest))
    # Multiple test correction
    if not (isinstance(multtest, type(None))):
        # Make big row with all pvalues
        tmpP=adjmat.values.ravel()
        # Find not nans
        I=~np.isnan(tmpP)
        Padj=np.zeros_like(tmpP)*np.nan
        # Do multiple test correction on only the tested ones
        Padj[I]=multitest.multipletests(tmpP[I], method=multtest)[1]
        # Rebuild adjmatrix
        adjmat = pd.DataFrame(data=Padj.reshape(adjmat.shape), columns=adjmat.columns, index=adjmat.index)
    
    # Return
    return(adjmat)

#%% Do multiple test correction
def multipletestcorrection(out, multtest, verbose=3):
    if verbose>=3: print('[HNET] Multiple test correction using %s' %(multtest))
    # Always do a multiple test correction but do not use it in the filtering step if not desired
    
    if out!=[]:
        # Get pvalues
        Praw=np.array(list(map(lambda x: x['P'], out)))
        I=np.isnan(Praw)
        Praw[I]=1

        # Multiple test correction
        if (isinstance(multtest, type(None))): 
            Padj=Praw
        else:
            #Padj=np.zeros_like(Praw)*np.nan
            Padj=multitest.multipletests(Praw, method=multtest)[1]
        
        for i in range(0,len(out)):
            out[i].update({'Padj':Padj[i]})
    
    return(out)

#%% Setup columns in correct dtypes
def filter_significance(out, alpha, multtest):
    if isinstance(multtest, type(None)):
        idx=np.where(np.array(list(map(lambda x: x['P']<=alpha, out))))[0]
    else:
        idx=np.where(np.array(list(map(lambda x: x['Padj']<=alpha, out))))[0]
    
    outf = [out[i] for i in idx]
    if outf==[]: outf=None
    return(outf)

#%% Cleaning
def nancleaning(datac, y):
    I     = datac.replace([np.inf, -np.inf, None, 'nan', 'None', 'inf', '-inf'], np.nan).notnull()
    datac = datac[I]
    yc    = y[I]
    return(datac, yc)

#%% Add combinations
def make_n_combinations(Xhot, Xlabx, combK, y_min, verbose=3):
    Xlabo=Xlabx.copy()
    if isinstance(y_min, type(None)): y_min=1
    # If any, run over combinations
    if not isinstance(combK, type(None)) and combK>1:
        out_hot  = Xhot
        out_labo = Xlabo
        out_labx = list(map(lambda x: [x], Xlabx))
        # Run over all combinations
        for k in np.arange(2,combK+1):
            # Make smart combinations because of mutual exclusive classes
            [cmbn_hot, cmbn_labX, cmbn_labH, cmbn_labO] = cmbnN(Xhot, Xlabx, y_min, k)
            # If any combinations is found, add to dataframe
            if len(cmbn_labX)>0:
                if verbose>=3: print('[HNET] Adding %d none mutual exclusive combinations with k=[%d] features.' %(cmbn_hot.shape[1], k))
                out_hot  = pd.concat([out_hot, pd.DataFrame(data=cmbn_hot, columns=cmbn_labH).astype(int)], axis=1)
                out_labo = np.append(out_labo, cmbn_labO, axis=0)
                out_labx = out_labx+cmbn_labX
            else:
                if verbose>=3: print('[HNET] No combinatorial features detected with k=[%d] features. No need to search for higher k.' %(k))
                break
        
        # Add to one-hot dataframe
        Xhot  = out_hot
        Xlabo = out_labo
        Xlabx = out_labx
    
    assert Xhot.shape[1]==len(Xlabx), print('one-hot matrix should have equal size with xlabels')
    assert Xhot.shape[1]==len(Xlabo), print('one-hot matrix should have equal size with olabels')
    return(Xhot,Xlabx,Xlabo)
    
#%% Add combinations
def cmbnN(Xhot, Xlabx, y_min, k):
    # Take only those varibles if combinations is larger then N (otherwise it is not mutually exclusive)
    [uilabx, uicount]=np.unique(Xlabx, return_counts=True)
    I=np.isin(Xlabx, uilabx[uicount>k])

    # cmnb_labx = np.array(list(itertools.combinations(Xhot.columns[I], k)))
    cmbn_idx  = np.array(list(itertools.combinations(np.where(I)[0], k)))
    cmbn_hot  = []
    cmbn_labH = []
    cmbn_labX = []
    cmbn_labO = []
    for idx in cmbn_idx:
        # Compute product
        prodFeat = Xhot.iloc[:,idx].prod(axis=1)
        # Store if allowed
        if sum(prodFeat)>=y_min:
            cmbn_hot.append(prodFeat.values)
            cmbn_labH.append('_&_'.join(Xhot.columns[idx]))
            cmbn_labX.append(list(np.unique(Xlabx[idx])))
            cmbn_labO.append('_&_'.join(np.unique(Xlabx[idx])))

    # Make array
    cmbn_hot=np.array(cmbn_hot).T
    # Combine repetative values
    #assert cmbn_hot.shape[1]==len(cmbn_labX), print('one-hot matrix should have equal size with labels')
    return(cmbn_hot, cmbn_labX, cmbn_labH, cmbn_labO)

#%% Do the math
def do_the_math(df, X_comb, dtypes, X_labx, simmat_padj, simmat_labx, param, i):
    count=0
    out=[]
    # Get response variable to test association
    y=X_comb.iloc[:,i].values.astype(str)
    # Get column name
    colname=X_comb.columns[i]
    # Do something if response variable has more then 1 option. 
    if len(np.unique(y))>1:
        if param['verbose']>=4: print('[HNET] Working on [%s]' %(X_comb.columns[i]), end='')
        # Remove columns if it belongs to the same categorical subgroup; these can never overlap!
        I=~np.isin(df.columns, X_labx[i])
        # Compute fit
        dfout=enrichment(df.loc[:,I], y, y_min=param['y_min'], alpha=1, multtest=None, dtypes=dtypes[I], specificity=param['specificity'], verbose=0)
        # Count        
        count=count+dfout.shape[0]
        # Match with dataframe and store
        if not dfout.empty:
            # Column names
            idx           = np.where(dfout['category_label'].isna())[0]
            catnames      = dfout['category_name']
            colnames      = catnames+'_'+dfout['category_label']
            colnames[idx] = catnames[idx].values
            # Add new column and index
            [simmat_padj, simmat_labx]=addcolumns(simmat_padj, colnames, simmat_labx, catnames)
            # Store values
            [IA,IB]=ismember(simmat_padj.index.values.astype(str), colnames.values.astype(str))
            simmat_padj.loc[colname, IA] = dfout['Padj'].iloc[IB].values
            # Count nr. successes
            out = [colname, X_comb.iloc[:,i].sum()/X_comb.shape[0]]
            # showprogress
            if param['verbose']>=4: print('[%g]' %(len(IB)), end='')
    else:
        if param['verbose']>=4: print('[HNET] Skipping [%s] because length of unique values=1' %(X_comb.columns[i]), end='')

    if param['verbose']>=4: print('')
    # Return
    return(out, simmat_padj, simmat_labx)

#%% Do the math
def post_processing(simmat_padj, nr_succes_pop_n, simmat_labx, param):
    # Clean label names
    simmat_padj.columns=list(map(lambda x: x[:-2] if x[-2:]=='.0' else x, simmat_padj.columns))
    simmat_padj.index=list(map(lambda x: x[:-2] if x[-2:]=='.0' else x, simmat_padj.index.values))
    nr_succes_pop_n=np.array(nr_succes_pop_n)
    nr_succes_pop_n[:,0]=list(map(lambda x: x[:-2] if x[-2:]=='.0' else x,  nr_succes_pop_n[:,0]))

    # Multiple test correction
    simmat_padj = multipletestcorrectionAdjmat(simmat_padj, param['multtest'], verbose=param['verbose'])
    # Remove variables for which both rows and columns are empty
    if param['dropna']: [simmat_padj, simmat_labx]=drop_empty(simmat_padj, simmat_labx, verbose=param['verbose'])
    # Fill empty fields
    if param['fillna']: simmat_padj.fillna(1, inplace=True)
    # Remove those with P>alpha, to prevent unnecesarilly edges
    simmat_padj[simmat_padj>param['alpha']]=1
    # Convert P-values to -log10 scale
    adjmatLog = logscale(simmat_padj)
    
    # Remove edges from matrix
    if param['dropna']:
        idx1=np.where((simmat_padj==1).sum(axis=1)==simmat_padj.shape[0])[0]
        idx2=np.where((simmat_padj==1).sum(axis=0)==simmat_padj.shape[0])[0]
        keepidx= np.setdiff1d(np.arange(simmat_padj.shape[0]), np.intersect1d(idx1,idx2))
        simmat_padj=simmat_padj.iloc[keepidx,keepidx]
        adjmatLog=adjmatLog.iloc[keepidx,keepidx]
        simmat_labx=simmat_labx[keepidx]
        [IA,_]=ismember(nr_succes_pop_n[:,0], simmat_padj.columns.values)
        nr_succes_pop_n=nr_succes_pop_n[IA,:]
    
    return(simmat_padj, nr_succes_pop_n, adjmatLog, simmat_labx)

#%% Scale weights
def scale_weights(weights, node_size_limits):
    out = MinMaxScaler(feature_range=(node_size_limits[0],node_size_limits[1])).fit_transform(np.append('0',weights).astype(float).reshape(-1,1)).flatten()[1:]
    return(out)

#%% Split filepath
def path_split(filepath, rem_spaces=False):
    [dirpath, filename]=os.path.split(filepath)
    [filename,ext]=os.path.splitext(filename)
    if rem_spaces:
        filename=filename.replace(' ','_')
    return(dirpath, filename, ext)

#%% From savepath to full and correct path
def path_correct(savepath, filename='fig', ext='.png'):
    '''
    savepath can be a string that looks like below.
    savepath='./tmp'
    savepath='./tmp/fig'
    savepath='./tmp/fig.png'
    savepath='fig.png'
    '''
    out=None
    if not isinstance(savepath, type(None)):
        # Set savepath and filename
        [getdir,getfile,getext]=path_split(savepath)
        # Make savepath
        if len(getdir[1:])==0: getdir=''
        if len(getfile)==0: getfile=filename
        if len(getext)==0: getext=ext
        # Create dir
        if len(getdir)>0:
            path=Path(getdir)
            path.mkdir(parents=True, exist_ok=True)
        # Make final path
        out=os.path.join(getdir,getfile+getext)
    return(out)

#%% Make network d3
def plot_d3graph(out, node_size_limits=[6,15], savepath=None, node_color=None, directed=True, showfig=True):
    [IA,IB]=ismember(out['simmatLogP'].columns, out['counts'][:,0])
    node_size = np.repeat(node_size_limits[0], len(out['simmatLogP'].columns))
    node_size[IA]=scale_weights(out['counts'][IB,1], node_size_limits)

    # Color node using network-clustering
    if node_color=='cluster':
        _,labx=plot_network(out, showfig=False)
    else:
        labx = label_encoder.fit_transform(out['labx'])

    # Make network
    #if directed:
    #    simmatLogP = out['simmatLogP'].copy()>0
    #else:
        # Make symmetric
    simmatLogP = to_symmetric(out, make_symmetric='logp')
    
    # Make network
    Gout = d3graph(simmatLogP.T, path=savepath, node_size=node_size, charge=500,  width=1500, height=800, collision=0.1, node_color=labx, directed=directed, showfig=showfig)
    
    # Return
    Gout['labx']=labx
    return(Gout)

#%% Make network d3
def plot_network(out, scale=2, dist_between_nodes=0.4, node_size_limits=[25,500], node_color=None, showfig=True, savepath=None, figsize=[15,10], pos=None, layout='fruchterman_reingold', dpi=250):
    config=dict()
    config['scale']=scale
    config['node_color']=node_color
    config['dist_between_nodes']=dist_between_nodes # k=0.4
    config['node_size_limits']=node_size_limits
    config['layout']=layout
    config['iterations']=50
    config['dpi']=dpi

    # Set savepath and filename
    config['savepath']=path_correct(savepath, filename='hnet_network', ext='.png')
    
    # Get adjmat
    adjmatLog = out['simmatLogP'].copy()

    # Set weights for edges
    adjmatLogWEIGHT = adjmatLog.copy()
    np.fill_diagonal(adjmatLogWEIGHT.values, 0)
    adjmatLogWEIGHT = pd.DataFrame(index=adjmatLog.index.values, data=MinMaxScaler(feature_range=(0,20)).fit_transform(adjmatLogWEIGHT), columns=adjmatLog.columns)
    
    # Set size for node 
    [IA,IB]=ismember(out['simmatLogP'].columns, out['counts'][:,0])
    node_size = np.repeat(node_size_limits[0], len(out['simmatLogP'].columns))
    node_size[IA]=scale_weights(out['counts'][IB,1], node_size_limits)

    # Make new graph (G) and add properties to nodes
    G = nx.DiGraph(directed=True)
    for i in range(0, adjmatLog.shape[0]):
        G.add_node(adjmatLog.index.values[i], node_size=node_size[i], node_label=out['labx'][i])
    # Add properties to edges
    np.fill_diagonal(adjmatLog.values, 0)
    for i in range(0, adjmatLog.shape[0]):
        idx=np.where(adjmatLog.iloc[i,:]>0)[0]
        labels=adjmatLog.iloc[i,idx]
        labelsWEIGHT=adjmatLogWEIGHT.iloc[i,idx]
        
        for k in range(0,len(labels)):
            G.add_edge(labels.index[k], adjmatLog.index[i], weight=labels.values[k].astype(int), capacity=labelsWEIGHT.values[k])

    edges = G.edges()
    edge_weights = np.array([G[u][v]['weight'] for u,v in edges])
    edge_weights = MinMaxScaler(feature_range=(0.5,8)).fit_transform(edge_weights.reshape(-1,1)).flatten()

    # Cluster
    if node_color=='cluster':
        [_,labx]=network.cluster(G.to_undirected())
    else:
        labx = label_encoder.fit_transform(out['labx'])
    
    # G = nx.DiGraph() # Directed graph
    # Layout
    if isinstance(pos, type(None)):
        pos = nx.fruchterman_reingold_layout(G, weight='edge_weights', k=config['dist_between_nodes'], scale=config['scale'], iterations=config['iterations'])
    else:
        pos = network.graphlayout(G, pos=pos, scale=config['scale'], layout=config['layout'])

    # pos = nx.spring_layout(G, weight='edge_weights', k=config['dist_between_nodes'], scale=config['scale'], iterations=config['iterations'])

    # Boot figure
    if showfig or (not isinstance(config['savepath'], type(None))):
        # [fig,ax]=plt.figure(figsize=(figsize[0],figsize[1]))
        [fig,ax]=plt.subplots(figsize=(figsize[0],figsize[1]))
        options = {
        # 'node_color': 'grey',
        'arrowsize': 12,
        'font_size':18,
        'font_color':'black',
        }
        # Draw plot
        nx.draw(G, pos, with_labels=True, **options, node_size=node_size*5, width=edge_weights, node_color=labx, cmap='Paired')
        # Plot weights
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G,'weight'))
        # Figure cleaning
        ax = plt.gca()
        ax.set_axis_off()
        # Show figure
        if showfig: plt.show()
        # Save figure to disk
        if not isinstance(config['savepath'], type(None)):
            savefig(fig, config['savepath'], dpi=config['dpi'], transp=True)

    
    Gout=dict()
    Gout['G']=G
    Gout['labx']=labx
    Gout['pos']=pos
    
    return(Gout)

# %% Tempdir
def tempdir(savepath):
    if savepath is None:
        savepath=os.path.join(tempfile.gettempdir(),'')
    return(savepath)

#%% Make plot of the structure_learning
def plot_heatmap(out, cluster=False, figsize=[15,10], savepath=None, verbose=3):
    simmatP = out['simmatP'].copy()
    adjmatLog = out['simmatLogP'].copy()
    # Set savepath and filename
    savepath = path_correct(savepath, filename='hnet_heatmap', ext='.png')

    try:
        if cluster==False:
            np.fill_diagonal(simmatP.values, 0)
            np.fill_diagonal(adjmatLog.values, np.maximum(1, np.max(adjmatLog.values)))
            # Add additional lines to avoid distortion of the heatmap
            # simmatP=pd.concat([pd.DataFrame(data=np.ones((1,simmatP.shape[1]))*np.nan, columns=simmatP.columns), simmatP])
            # simmatP=simmatP.T
            # simmatP['']=np.nan
            # simmatP=simmatP.T
            # adjmatLog=pd.concat([pd.DataFrame(data=np.zeros((1,adjmatLog.shape[1]))*np.nan, columns=adjmatLog.columns), adjmatLog])
            # adjmatLog=adjmatLog.T
            # adjmatLog['']=np.nan
            # adjmatLog=adjmatLog.T


        savepath1=''
        savepath2=''
        if savepath is not None:
            [getdir,getname,getext]=path_split(savepath)
            if getname=='': getname='heatmap'
            savepath1=os.path.join(getdir,getname+'_P'+getext)
            savepath2=os.path.join(getdir,getname+'_logP'+getext)

        if cluster:
            fig1=imagesc.cluster(simmatP.fillna(value=0).values, row_labels=simmatP.index.values, col_labels=simmatP.columns.values, cmap='Reds', figsize=figsize)
            fig2=imagesc.cluster(adjmatLog.fillna(value=0).values, row_labels=adjmatLog.index.values, col_labels=adjmatLog.columns.values, cmap='Reds', figsize=figsize)
            # savepath=savepath2
        else:
            fig1=imagesc.plot(simmatP.fillna(value=0).values, row_labels=simmatP.index.values, col_labels=simmatP.columns.values, cmap='Reds', figsize=figsize)
            fig2=imagesc.plot(adjmatLog.fillna(value=0).values, row_labels=adjmatLog.index.values, col_labels=adjmatLog.columns.values, cmap='Reds', figsize=figsize)
        
        if savepath is not None:
            if verbose>=3: print('[HNET.plot_heatmap] Saving figure..')
            _ = savefig(fig1, savepath1, transp=True)
            _ = savefig(fig2, savepath2, transp=True)
        
    except:
        print('[HNET][plot_heatmap] Failed making imagesc plot.')

#%% Make adjacency matrix symmetric with repect to the diagonal
def to_symmetric(out, make_symmetric='logp', verbose=3):
    assert np.sum(np.isin(out['simmatLogP'].index.values, out['simmatLogP'].columns.values))==np.max(out['simmatLogP'].shape), 'Adjacency matrix must have similar number of rows and columns! Re-run HNet with dropna=False!'
    progressbar=(True if verbose==0 else False)
    columns=out['simmatLogP'].columns.values
    index=out['simmatLogP'].index.values
    
    # Make selected matrix symmetric
    if make_symmetric=='logp':
        adjmat=out['simmatLogP'].values.copy()
    else:
        adjmat=out['simmatP'].values.copy()
    
    # Make output matrix
    adjmatS=np.zeros(adjmat.shape, dtype=float)
    
    # Make symmetric using maximum as combining function
    for i in tqdm(range(adjmat.shape[0]), disable=progressbar):
        for j in range(adjmat.shape[1]):
            if make_symmetric=='logp':
                # Maximum -log10(P) values
                score = np.maximum(adjmat[i,j], adjmat[j,i])
            else:
                # Minimum P-values
                score = np.minimum(adjmat[i,j], adjmat[j,i])
                
            adjmatS[i,j] = score
            adjmatS[j,i] = score
    
    # Make dataframe and return
    adjmatS=pd.DataFrame(index=index, data=adjmatS, columns=columns, dtype=float)
    return(adjmatS)

#%% Comparison of two networks
def compare_networks(adjmat_true, adjmat_pred, pos=None, showfig=True, width=15, height=8, verbose=3):
    [scores, adjmat_diff] = network.compare_networks(adjmat_true, adjmat_pred, pos=pos, showfig=showfig, width=width, height=height, verbose=verbose)
    return(scores, adjmat_diff)

#%% Example data
def import_example(getfile='titanic'):
    '''
    
    Parameters
    ----------
    getfile : String, optional
        'titanic'
        'sprinkler'

    Returns
    -------
    df : DataFrame

    '''
    
    if getfile=='titanic':
        getfile='titanic_train.zip'
    else:
        getfile='sprinkler.zip'

    print('[HNET] Loading %s..' %getfile)
    curpath = os.path.dirname(os.path.abspath( __file__ ))
    PATH_TO_DATA=os.path.join(curpath,'data',getfile)
    if os.path.isfile(PATH_TO_DATA):
        df=pd.read_csv(PATH_TO_DATA, sep=',')
        return df
    else:
        print('[HNET] Oops! Example data not found!')
        return None

#%% Main
#if __name__ == "__main__":
#    parser = argparse.ArgumentParser()
#    args = parser.parse_args()
#    fit(**vars(args))