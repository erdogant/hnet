import df2onehot as df2onehot
from ismember import ismember

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, ranksums
import statsmodels.stats.multitest as multitest
from sklearn.preprocessing import MinMaxScaler

import os
import tempfile
from pathlib import Path
import itertools

# %% Compute significance
def _compute_significance(df, y, dtypes, specificity=None, verbose=3):
    out=[]
    # Run over all columns
    for i in range(0, df.shape[1]):
        if (i>0) and (verbose>=3): print('')
        if verbose>=3: print('[hnet] >Analyzing [%s] %s' %(dtypes[i], df.columns[i]), end='')
        colname = df.columns[i]

        # Clean nan fields
        [datac, yc] = _nancleaning(df[colname], y)
        # In a two class model, remove 0-catagory
        uiy = np.unique(yc)
        # No need to compute _other_ because it is a mixed group that is auto generated based on y_min
        uiy=uiy[uiy!='_other_']

        if len(uiy)==1 and (uiy=='0'):
            if verbose>=4: print('[hnet] >The response variable [y] has only one catagory; [0] which is seen as the negative class and thus ignored.')
            uiy=uiy[uiy!='0']

        if len(uiy)==2:
            if verbose>=4: print('[hnet] >The response variable [y] has two catagories, the catagory 0 is seen as the negative class and thus ignored.')
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
                if (datacOnehot.shape[1]==2) & (np.any(np.isin(datacOnehot.columns, '0.0'))):
                    datacOnehot.drop(labels=['0.0'], axis=1, inplace=True)
                # Run over all unique entities/cats in column for target vlue
                for k in range(0, datacOnehot.shape[1]):
                    outtest = _prob_hypergeo(datacOnehot.iloc[:, k], yc==target)
                    outtest.update({'y': target})
                    outtest.update({'category_name': colname})
                    out.append(outtest)

            # Numerical
            if dtypes[i]=='num':
                outtest = _prob_ranksums(datac, yc==target, specificity=specificity)
                outtest.update({'y': target})
                outtest.update({'category_name': colname})
                out.append(outtest)
            # Print dots
            if verbose>=3: print('.', end='')

    if verbose>=3: print('')
    return(out)


# %% Wilcoxon Ranksum test
def _prob_ranksums(datac, yc, specificity=None):
    P=np.nan
    zscore=np.nan
    datac=datac.values
    getsign=''

    # Wilcoxon Ranksum test
    if sum(yc==True)>1 and sum(yc==False)>1:
        [zscore, P] = ranksums(datac[yc==True], datac[yc==False])

    # Store
    out=dict()
    out['P']=P
    out['logP']=np.log(P)
    out['zscore']=zscore
    out['popsize_M']=len(yc)
    out['nr_succes_pop_n']=np.sum(yc == True)
    out['nr_not_succes_pop_n']=np.sum(yc == False)
    out['dtype']='numerical'

    if np.isnan(zscore) is False and np.sign(zscore)>0:
        getsign='high_'
    else:
        getsign='low_'

    if specificity=='low':
        out['category_label']=getsign[:-1]
    elif specificity=='medium':
        out['category_label']=getsign + str(('%.1f' %(np.median(datac[yc==True]))))
    elif specificity=='high':
        out['category_label']=getsign + str(('%.3f' %(np.median(datac[yc==True]))))
    else:
        out['category_label']=''

    return(out)


# %% Hypergeometric test
def _prob_hypergeo(datac, yc):
    """Compute hypergeometric Pvalue.

    Description
    -----------
    Suppose you have a lot of 100 floppy disks (M), and you know that 20 of them are defective (n).
    What is the prbability of drawing zero to 2 floppy disks (N=2), if you select 10 at random (N).
    P=hypergeom.sf(2,100,20,10)

    """
    P = np.nan
    logP = np.nan
    M = len(yc)  # Population size: Total number of samples, eg total number of genes; 10000
    n = np.sum(datac)  # Number of successes in population, known in pathway, eg 2000
    N = np.sum(yc)  # sample size: Random variate, eg clustersize or groupsize, over expressed genes, eg 300
    X = np.sum(np.logical_and(yc, datac)) - 1  # Let op, de -1 is belangrijk omdatje P<X wilt weten ipv P<=X. Als je P<=X doet dan kan je vele false positives krijgen als bijvoorbeeld X=1 en n=1 oid

    # Test
    if np.any(yc) and (X>0):
        P = hypergeom.sf(X, M, n, N)
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

# %% Make logscale
def _logscale(simmat_padj):
    # Set minimum amount
    if np.any(~np.isnan(simmat_padj.values.ravel())):
        simmat_padj[simmat_padj==0]=1e-323
        adjmatLog=(-np.log10(simmat_padj)).copy()
        adjmatLog[adjmatLog == -np.inf] = np.nanmax(adjmatLog[adjmatLog != np.inf])
        adjmatLog[adjmatLog == np.inf] = np.nanmax(adjmatLog[adjmatLog != np.inf])
        adjmatLog[adjmatLog == -0] = 0
    else:
        adjmatLog = simmat_padj.copy()
    return(adjmatLog)

# %% Do multiple test correction
def _multipletestcorrectionAdjmat(adjmat, multtest, verbose=3):
    if verbose>=3: print('[hnet] >Multiple test correction using %s' %(multtest))
    # Multiple test correction
    if multtest is not None:
        # Make big row with all pvalues
        tmpP = adjmat.values.ravel()
        # Find not nans
        Iloc = ~np.isnan(tmpP)
        Padj = np.zeros_like(tmpP) * np.nan
        # Do multiple test correction on only the tested ones
        if np.any(Iloc):
            Padj[Iloc]=multitest.multipletests(tmpP[Iloc], method=multtest)[1]
        # Rebuild adjmatrix
        adjmat = pd.DataFrame(data=Padj.reshape(adjmat.shape), columns=adjmat.columns, index=adjmat.index)

    # Return
    return(adjmat)

# %% Do multiple test correction
def _multipletestcorrection(out, multtest, verbose=3):
    # Always do a multiple test correction but do not use it in the filtering step if not desired
    if verbose>=3: print('[hnet] >Multiple test correction using %s' %(multtest))

    if out!=[]:
        # Get pvalues
        Praw = np.array(list(map(lambda x: x['P'], out)))
        Iloc = np.isnan(Praw)
        Praw[Iloc] = 1

        # Multiple test correction
        if multtest is None:
            Padj = Praw
        else:
            # Padj=np.zeros_like(Praw)*np.nan
            Padj=multitest.multipletests(Praw, method=multtest)[1]

        for i in range(0, len(out)):
            out[i].update({'Padj':Padj[i]})

    return(out)

# %% Add combinations
def _make_n_combinations(Xhot, Xlabx, combK, y_min, verbose=3):
    Xlabo=Xlabx.copy()
    if isinstance(y_min, type(None)): y_min=1
    # If any, run over combinations
    if not isinstance(combK, type(None)) and combK>1:
        out_hot = Xhot
        out_labo = Xlabo
        out_labx = list(map(lambda x: [x], Xlabx))
        # Run over all combinations
        for k in np.arange(2,combK + 1):
            # Make smart combinations because of mutual exclusive classes
            [cmbn_hot, cmbn_labX, cmbn_labH, cmbn_labO] = _cmbnN(Xhot, Xlabx, y_min, k)
            # If any combinations is found, add to dataframe
            if len(cmbn_labX)>0:
                if verbose>=3: print('[hnet] >Adding %d none mutual exclusive combinations with k=[%d] features.' %(cmbn_hot.shape[1], k))
                out_hot = pd.concat([out_hot, pd.DataFrame(data=cmbn_hot, columns=cmbn_labH).astype(int)], axis=1)
                out_labo = np.append(out_labo, cmbn_labO, axis=0)
                out_labx = out_labx + cmbn_labX
            else:
                if verbose>=3: print('[hnet] >No combinatorial features detected with k=[%d] features. No need to search for higher k.' %(k))
                break

        # Add to one-hot dataframe
        Xhot = out_hot
        Xlabo = out_labo
        Xlabx = out_labx

    assert Xhot.shape[1]==len(Xlabx), print('one-hot matrix should have equal size with xlabels')
    assert Xhot.shape[1]==len(Xlabo), print('one-hot matrix should have equal size with olabels')
    return(Xhot,Xlabx,Xlabo)


# %% Add combinations
def _cmbnN(Xhot, Xlabx, y_min, k):
    # Take only those varibles if combinations is larger then N (otherwise it is not mutually exclusive)
    [uilabx, uicount]=np.unique(Xlabx, return_counts=True)
    Iloc = np.isin(Xlabx, uilabx[uicount>k])

    # cmnb_labx = np.array(list(itertools.combinations(Xhot.columns[I], k)))
    cmbn_idx = np.array(list(itertools.combinations(np.where(Iloc)[0], k)))
    cmbn_hot = []
    cmbn_labH = []
    cmbn_labX = []
    cmbn_labO = []
    for idx in cmbn_idx:
        # Compute product
        prodFeat = Xhot.iloc[:, idx].prod(axis=1)
        # Store if allowed
        if sum(prodFeat)>=y_min:
            cmbn_hot.append(prodFeat.values)
            cmbn_labH.append('_&_'.join(Xhot.columns[idx]))
            cmbn_labX.append(list(np.unique(Xlabx[idx])))
            cmbn_labO.append('_&_'.join(np.unique(Xlabx[idx])))

    # Make array
    cmbn_hot=np.array(cmbn_hot).T
    # Combine repetative values
    # assert cmbn_hot.shape[1]==len(cmbn_labX), print('one-hot matrix should have equal size with labels')
    return(cmbn_hot, cmbn_labX, cmbn_labH, cmbn_labO)


# %% Add columns
def _addcolumns(simmat_padj, colnames, Xlabx, catnames):
    Iloc = np.isin(colnames.values.astype(str), simmat_padj.index.values)
    if np.any(Iloc):
        newcols=list((colnames.values[Iloc == False]).astype(str))
        newcats=list((catnames[Iloc == False]).astype(str))

        # Make new columns in dataframe
        for col, cat in zip(newcols, newcats):
            simmat_padj[col]=np.nan
            Xlabx = np.append(Xlabx, cat)

        addrow=pd.DataFrame(index=newcols, columns=simmat_padj.columns.values).astype(float)
        simmat_padj=pd.concat([simmat_padj, addrow])
    return(simmat_padj, Xlabx)


# %% Remove columns without dtype
def _remove_columns_without_dtype(df, dtypes, verbose=3):
    if not isinstance(dtypes, str):
        assert df.shape[1]==len(dtypes), 'Columns in df and dtypes should match! [hnet.remove_columns_without_dtype]'
        Iloc = np.isin(dtypes, '')
        if np.any(Iloc):
            remcols=df.columns[Iloc].values
            df.drop(columns=remcols, inplace=True)
            dtypes=list(np.array(dtypes)[(Iloc==False)])
            if verbose>=3: print('[hnet] >%.0f columns are removed.' %(len(remcols)))

        assert df.shape[1]==len(dtypes), 'Columns in df and dtypes should match! [hnet.remove_columns_without_dtype]'

    return(df, dtypes)


# %% Clean empty rows
def _drop_empty(df, Xlabx, verbose=3):
    dfO=df.copy()
    cols=dfO.columns.values
    rows=dfO.index.values

    # Set diagonal on nan
    np.fill_diagonal(df.values, np.nan)

    droplabel=[]
    for col in cols:
        if np.any(cols==col):
            if np.all(np.logical_and(df.loc[:, cols==col].isna().values.reshape(-1, 1), df.loc[rows==col, :].isna().values.reshape(-1, 1))):
                if verbose>=3: print('[hnet] >Dropping %s' %(col))
                droplabel.append(col)

    # Remove labels from the original df
    Xlabx=Xlabx[np.isin(dfO.columns, droplabel)==False]
    dfO.drop(labels=droplabel, axis=0, inplace=True)
    dfO.drop(labels=droplabel, axis=1, inplace=True)

    return(dfO, Xlabx)

# %% Setup columns in correct dtypes
def _filter_significance(out, alpha, multtest):
    if isinstance(multtest, type(None)):
        idx=np.where(np.array(list(map(lambda x: x['P']<=alpha, out))))[0]
    else:
        idx=np.where(np.array(list(map(lambda x: x['Padj']<=alpha, out))))[0]

    outf = [out[i] for i in idx]
    if outf==[]: outf=None
    return(outf)


# %% Cleaning
def _nancleaning(datac, y):
    Iloc = datac.replace([np.inf, -np.inf, None, 'nan', 'None', 'inf', '-inf'], np.nan).notnull()
    datac = datac[Iloc]
    yc = y[Iloc]
    return(datac, yc)


# %% Do the math
def _post_processing(simmat_padj, nr_succes_pop_n, simmat_labx, alpha, multtest, fillna, dropna, verbose=3):
    # Clean label names by chaning X.0 into X
    simmat_padj.columns=list(map(lambda x: x[:-2] if x[-2:]=='.0' else x, simmat_padj.columns))
    simmat_padj.index=list(map(lambda x: x[:-2] if x[-2:]=='.0' else x, simmat_padj.index.values))
    nr_succes_pop_n=np.array(nr_succes_pop_n)
    nr_succes_pop_n[:,0]=list(map(lambda x: x[:-2] if x[-2:]=='.0' else x, nr_succes_pop_n[:,0]))

    if verbose>=5: print(simmat_padj)
    # Multiple test correction
    simmat_padj = _multipletestcorrectionAdjmat(simmat_padj, multtest, verbose=verbose)
    # Remove variables for which both rows and columns are empty
    if dropna: [simmat_padj, simmat_labx]=_drop_empty(simmat_padj, simmat_labx, verbose=verbose)
    # Fill empty fields
    if fillna: simmat_padj.fillna(1, inplace=True)
    # Remove those with P>alpha, to prevent unnecesarilly edges
    simmat_padj[simmat_padj>alpha]=1
    # Convert P-values to -log10 scale
    adjmatLog = _logscale(simmat_padj)

    # Set zeros on diagonal but make sure it is correctly ordered
    if np.all(adjmatLog.index.values==adjmatLog.columns.values):
        np.fill_diagonal(adjmatLog.values, 0)
    if np.all(simmat_padj.index.values==simmat_padj.columns.values):
        np.fill_diagonal(simmat_padj.values, 1)

    # Remove edges from matrix
    if dropna:
        idx1=np.where((simmat_padj==1).sum(axis=1)==simmat_padj.shape[0])[0]
        idx2=np.where((simmat_padj==1).sum(axis=0)==simmat_padj.shape[0])[0]
        keepidx= np.setdiff1d(np.arange(simmat_padj.shape[0]), np.intersect1d(idx1,idx2))
        simmat_padj=simmat_padj.iloc[keepidx,keepidx]
        adjmatLog=adjmatLog.iloc[keepidx,keepidx]
        simmat_labx=simmat_labx[keepidx]
        [IA,_]=ismember(nr_succes_pop_n[:,0], simmat_padj.columns.values)
        nr_succes_pop_n=nr_succes_pop_n[IA,:]

    return(simmat_padj, adjmatLog, simmat_labx, nr_succes_pop_n)


# %% Scale weights
def _scale_weights(weights, node_size_limits):
    out = MinMaxScaler(feature_range=(node_size_limits[0],node_size_limits[1])).fit_transform(np.append('0',weights).astype(float).reshape(-1,1)).flatten()[1:]
    return(out)


# %% Split filepath
def _path_split(filepath, rem_spaces=False):
    [dirpath, filename]=os.path.split(filepath)
    [filename,ext]=os.path.splitext(filename)
    if rem_spaces:
        filename=filename.replace(' ','_')
    return(dirpath, filename, ext)


# %% From savepath to full and correct path
def _path_correct(savepath, filename='fig', ext='.png'):
    """Correcth the path for filename.

    Description
    -----------
    savepath can be a string that looks like below.
    savepath='./tmp'
    savepath='./tmp/fig'
    savepath='./tmp/fig.png'
    savepath='fig.png'
    
    """
    out=None
    if not isinstance(savepath, type(None)):
        # Set savepath and filename
        [getdir, getfile, getext] = _path_split(savepath)
        # Make savepath
        if len(getdir[1:])==0: getdir=''
        if len(getfile)==0: getfile=filename
        if len(getext)==0: getext=ext
        # Create dir
        if len(getdir)>0:
            path=Path(getdir)
            path.mkdir(parents=True, exist_ok=True)
        # Make final path
        out=os.path.join(getdir,getfile + getext)
    return(out)


# %%
def _white_black_list(df, dtypes, white_list, black_list):
    # Keep only variables that are in white_list.
    if white_list is not None:
        white_list = [x.lower() for x in white_list]
        Iloc = np.isin(df.columns.str.lower(), white_list)
        df = df.loc[:,Iloc]

    # Exclude variables that are in black_list.
    if black_list is not None:
        black_list = [x.lower() for x in black_list]
        Iloc = ~np.isin(df.columns.str.lower(), black_list)
        df = df.loc[:,Iloc]
        # Remove also in dtypes
        if not isinstance(dtypes, str):
            if len(dtypes)!=len(Iloc):
                raise Exception('[hnet] >ERROR : dtypes should have same length as the input datafame! dtypes willbe automatically black listed too. Or do not specify dtypes.')
            else:
                dtypes = np.array(dtypes)[Iloc]

    # Check dtypes because things can have changed because of the white/black list
    if not isinstance(dtypes, str):
        if (white_list is not None) and len(dtypes)!=df.shape[1]:
            raise Exception('[hnet] >ERROR : dtypes should have same length as the white_list! Or do not specify dtypes.')
        if len(dtypes)!=df.shape[1]:
            raise Exception('[hnet] >ERROR : dtypes should have same length as your dataframe or white_list')
    
    if df.shape[1]<=1: raise Exception('[hnet] >ERROR : only %d variables are remaining. A minimum of 2 would be nice.' %(df.shape[1]))
    return df, dtypes


# %% Preprocessing
def _preprocessing(df, dtypes='pandas', y_min=10, perc_min_num=0.8, excl_background=None, white_list=None, black_list=None, verbose=3):
    df.columns = df.columns.astype(str)
    df.reset_index(drop=True, inplace=True)

    # Filter on white_list and black_list
    df, dtypes = _white_black_list(df, dtypes, white_list, black_list)
    # Remove columns without dtype
    [df, dtypes] = _remove_columns_without_dtype(df, dtypes, verbose=verbose)
    # Make onehot matrix for response variable y
    df_onehot = df2onehot.df2onehot(df, dtypes=dtypes, y_min=y_min, hot_only=True, perc_min_num=perc_min_num, excl_background=excl_background, verbose=verbose)
    dtypes = df_onehot['dtypes']

    # Make sure its limited to the number of y_min
    Iloc = (df_onehot['onehot'].sum(axis=0)>=y_min).values
    if np.any(Iloc==False):
        if verbose>=2: print('[hnet] >WARNING : Features with y_min needs another round of filtering. Fixing it now..')
        df_onehot['onehot']=df_onehot['onehot'].loc[:,Iloc]
        df_onehot['labx']=df_onehot['labx'][Iloc]

    # Some check before proceeding
    if (df_onehot['onehot'].empty) or (np.all(np.isin(dtypes, 'num'))): raise Exception('[hnet] >ALL data is excluded from the dataframe! There should be at least 1 categorical value!')
    if df.shape[1] != len(dtypes): raise Exception('[hnet] >DataFrame Shape and dtypes length does not match.')

    # Make all integer
    df_onehot['onehot'] = df_onehot['onehot'].astype(int)
    # Return
    return df, df_onehot, dtypes


# %% Tempdir
def _tempdir(savepath):
    if savepath is None:
        savepath = os.path.join(tempfile.gettempdir(), '')
    else:
        savepath = os.path.join(savepath, '')
    os.makedirs(savepath, exist_ok=True)

    return(savepath)


# %% Filter adjacency matrix
def _filter_adjmat(simmatLogP, threshold=None, min_edges=None, white_list=None, black_list=None, verbose=3):
    # Filter on white_list and black_list
    if min_edges is not None:
        simmatBOOL = simmatLogP.copy()>0
        if np.all(simmatBOOL.columns==simmatBOOL.index.values):
            Iloc = np.logical_or(simmatBOOL.sum(axis=0)>min_edges, simmatBOOL.sum(axis=1)>min_edges).values
            if verbose>=3: print('[hnet] >Filtering on edges: [%d] variables remain after filtering on a minimum of [%d] edges.' %(np.sum(Iloc), min_edges))
            simmatLogP = simmatLogP.loc[Iloc,Iloc]

    if threshold is not None:
        if verbose>=3: print('[hnet] >Filtering associations on threshold > %d' %(threshold))
        simmatLogP[simmatLogP<threshold]=0
    if white_list is not None:
        Irow = np.isin(simmatLogP.index.values, white_list)
        Icol = np.isin(simmatLogP.columns, white_list)
        if verbose>=3: print('[hnet] >Number of variables after white listing: [%d]' %(sum(Irow)))
        simmatLogP = simmatLogP.iloc[Irow,Icol]
    if black_list is not None:
        Irow = ~np.isin(simmatLogP.index.values, black_list)
        Icol = ~np.isin(simmatLogP.columns, black_list)
        if verbose>=3: print('[hnet] >Number of variables after black listing: [%d]' %(sum(Irow)))
        simmatLogP = simmatLogP.iloc[Irow, Icol]

    return simmatLogP

