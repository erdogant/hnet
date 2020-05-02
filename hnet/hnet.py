"""HNET: Graphical Hypergeometric-networks."""
# -------------------------------------------------
# Name        : hnet.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# github      : https://github.com/erdogant/hnet
# Licence     : See licences
# -------------------------------------------------

# %% Libraries
# Warnings
import warnings
warnings.filterwarnings("ignore")

# Custom packages
from d3graph import d3graph as d3graphs
from ismember import ismember
import imagesc
import df2onehot
# Local utils
from hnet.utils.savefig import savefig
import hnet.utils.network as network

# Known libraries
from scipy.stats import combine_pvalues, hypergeom, ranksums
import statsmodels.stats.multitest as multitest
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import networkx as nx
label_encoder = LabelEncoder()

# Internal
import wget
import os
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt


# %% Structure learning across all variables
class hnet():
    """HNET - Graphical Hypergeometric networks.

    Description
    -----------
    This is the main function to detect significant edge probabilities between pairs of vertices (node-links) given the input data set **df**.
    A multi-step process is performed which consisting 5 steps.

        1. Pre-processing: Typing and One-hot Enconding. Each feature is set as being categoric, numeric or is excluded. The typing can be user-defined or automatically determined on conditions. Encoding of features in a one-hot dense array is done for the categoric terms. The one-hot dense array is subsequently used to create combinatory features using k combinations over n features (without replacement).
        2. Combinations: Make smart combinations between features because many mutual exclusive classes do exists.
        3. Hypergeometric test: The final dense array is used to assess significance with the categoric features.
        4. Wilcoxon Ranksum: To assess significance across the numeric features (Xnumeric) in relation to the dense array (Xcombination), the Mann-Whitney-U test is performed.
        5. Multiple test correction: Declaring significance for node-links.

    The final output of HNet is an adjacency matrix containing edge weights that depicts the strength of pairs of vertices.
    The adjacency matrix can then be examined as a network representation using d3graph.

    Parameters
    ----------
    alpha : float [0..1], default : 0.05.
        Significance to keep only edges with <=alhpa.
        0.05 : (default)
        1    : (for all results)

    y_min : [Integer], default : 10.
        Minimum number of samples in a group. Should be [samples>=y_min]. All groups with less then y_min samples are labeled as _other_ and are not used in the model.
        10  (default)
        None

    k : int, [1..n] , default : 1.
        Number of combinatoric elements to create for the n features
        The default is 1.

    multtest : String, default : 'holm'.
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

    dtypes : list of str, default : 'pandas'.
        list strings, example: ['cat','num',''] of length y. By default the dtype is determined based on the pandas dataframe. Empty ones [''] are skipped.
        ['cat','cat','num','','cat',...]
        'pandas' (default)

    specificity : String, Configure how numerical data labels are stored. Setting this variable can be of use in the 'structure_learning' function for the creation of a network ([None] will glue most numerical labels together whereas [high] mostly will not).
            None : No additional information in the labels
            'low' : 'high' or 'low' are included that represents significantly higher or lower assocations compared to the rest-group.
            'medium' : (default) 'high' or 'low' are included with 1 decimal behind the comma.
            'high' : 'high' or 'low' are included with 3 decimal behind the comma.

    perc_min_num : float, Force column (int or float) to be numerical if unique non-zero values are above percentage.
        None
        0.8 (default)

    dropna : Bool, [True,False] Drop rows/columns in adjacency matrix that showed no significance
        True (default)
        False

    excl_background : String, String name to exclude from the background
        None (default)
        ['0.0']: To remove catagorical values with label 0

    verbose : Int, [0..5]. The higher the number, the more information is printed.
        0: No
        1: ERROR
        2: WARN
        3: INFO (default)
        4: DEBUG


    Returns
    -------
    dict
        The output is a dictionary containing the following keys:

    simmatP : pd.DataFrame()
        Adjacency matrix containing P-values between variable assocations.
    simmatLogP :  pd.DataFrame()
        -log10(P-value) of the simmatP.
    labx :  list of str
        Labels that are analyzed.
    dtypes :  list of str
        dtypes that are set for the labels.
    counts :  list of str
        Relative counts for the labels based on the number of successes in population.


    Examples
    --------
    >>> import hnet
    >>> df = hnet.import_example('sprinkler')
    >>> model = hnet.fit(df)
    >>> G = hnet.d3graph(model)
    """

    def __init__(self, alpha=0.05, y_min=10, k=1, multtest='holm', dtypes='pandas', specificity='medium', perc_min_num=0.8, dropna=True, excl_background=None):
        """Initialize distfit with user-defined parameters."""
        if (alpha is None): alpha=1
        self.alpha = alpha
        self.y_min = y_min
        self.k = k
        self.multtest = multtest
        self.dtypes = dtypes
        self.specificity = specificity
        self.perc_min_num = perc_min_num
        self.dropna = dropna
        self.fillna = True
        self.excl_background = excl_background

    def fit(self, df, verbose=3):
        """The fit() function is for learning model parameters from training data.
        """
        # Pre processing
        [df, df_onehot, dtypes] = _preprocessing(df, dtypes=self.dtypes, y_min=self.y_min, perc_min_num=self.perc_min_num, excl_background=self.excl_background, verbose=verbose)
        # Add combinations
        [X_comb, X_labx, X_labo] = _make_n_combinations(df_onehot['onehot'], df_onehot['labx'], self.k, self.y_min, verbose=verbose)
        # Print some
        if verbose>=3: print('[HNET] Structure learning across [%d] features.' %(X_comb.shape[1]))
        # Get numerical columns
        colNum = df.columns[df.dtypes == 'float64'].values
        simmat_labx = np.append(X_labo, colNum).astype(str)
        simmat_padj = pd.DataFrame(index=np.append(X_comb.columns, colNum).astype(str), columns=np.append(X_comb.columns, colNum).astype(str)).astype(float)
        # Return
        return df, simmat_padj, simmat_labx, X_comb, X_labx, dtypes


    def transform(self, df, simmat_padj, simmat_labx, X_comb, X_labx, dtypes, verbose=3):
        """The transform function applies the values of the parameters on the actual data and gives the normalized value.
        The fit_transform() function performs both in the same step. Note that the same value is got whether we perform in 2 steps or in a single step.
        """
        # Here we go! in parallel!
        # from multiprocessing import Pool
        # nr_succes_pop_n=[]
        # with Pool(processes=os.cpu_count()-1) as pool:
        #     for i in range(0,X_comb.shape[1]):
        #         result = pool.apply_async(_do_the_math, (df, X_comb, dtypes, X_labx, param, i,))
        #         nr_succes_pop_n.append(result)
        #     results = [result.get() for result in result_objs]
        #     print(len(results))

        count = 0
        nr_succes_pop_n = []
        for i in tqdm(range(0, X_comb.shape[1]), disable=(True if verbose==0 else False)):
            [nr_succes_i, simmat_padj, simmat_labx] = _do_the_math(df, X_comb, dtypes, X_labx, simmat_padj, simmat_labx, i, self.specificity, self.y_min, verbose=verbose)
            nr_succes_pop_n.append(nr_succes_i)
        # Message
        if verbose>=3: print('[HNET] Total number of computations: [%.0d]' %(count))
        # Post processing
        [simmat_padj, nr_succes_pop_n, adjmatLog, simmat_labx] = _post_processing(simmat_padj, nr_succes_pop_n, simmat_labx, self.alpha, self.multtest, self.fillna, self.dropna, verbose=3)
        # Return
        return simmat_padj, nr_succes_pop_n, adjmatLog, simmat_labx

    def fit_transform(self, df, return_dict=False, verbose=3):
        """Learn the structure in the data.

        Parameters
        ----------
        df : DataFrame, [NxM].
            N=rows->samples, and  M=columns->features.
           |    | f1| f2| f3|
           |----|---|---|---|
           | s1 | 0 | 0 | 1 |
           | s2 | 0 | 1 | 0 |
           | s3 | 1 | 1 | 0 |
        return_dict : bool : default : False.
            Return the results in a dictionary. Use this option if you want to export or save the results.
        verbose : int [1-5], default: 3
            Print information to screen. A higher number will print more.

        Returns
        -------
        dict.
        simmatP : pd.DataFrame()
            Adjacency matrix containing P-values between variable assocations.
        simmatLogP :  pd.DataFrame()
            -log10(P-value) of the simmatP.
        labx :  list of str
            Labels that are analyzed.
        dtypes :  list of str
            dtypes that are set for the labels.
        counts :  list of str
            Relative counts for the labels based on the number of successes in population.

        """
        assert isinstance(df, pd.DataFrame), 'Input data [df] must be of type pd.DataFrame()'

        df, simmat_padj, simmat_labx, X_comb, X_labx, dtypes = self.fit(df, verbose=verbose)
        # Here we go! Over all columns now
        simmat_padj, nr_succes_pop_n, adjmatLog, simmat_labx = self.transform(df, simmat_padj, simmat_labx, X_comb, X_labx, dtypes, verbose=verbose)
        # Store
        self.simmatP = simmat_padj
        self.simmatLogP = adjmatLog
        self.labx = simmat_labx.astype(str)
        self.dtypes = np.array(list(zip(df.columns.values.astype(str), dtypes)))
        self.counts = nr_succes_pop_n
        self.rules = self.combined_rules(verbose=0)

        # Use this option for storage of your model
        if return_dict:
            out = {}
            out['simmatP'] = self.simmatP
            out['simmatLogP'] = adjmatLog
            out['labx'] = self.labx
            out['dtypes'] = self.dtypes
            out['counts'] = self.counts
            out['rules'] = self.rules
            return(out)

    # Make network d3
    def d3graph(self, node_size_limits=[6,15], savepath=None, node_color=None, directed=True, showfig=True):
        """Interactive network creator.

        Description
        -----------
        This function creates a interactive and stand-alone network that is build on d3 javascript.
        d3graph is integrated into hnet and uses the -log10(P-value) adjacency matrix.
        Each column and index name represents a node whereas values >0 in the matrix represents an edge.
        Node links are build from rows to columns. Building the edges from row to columns only matters in directed cases.
        The network nodes and edges are adjusted in weight based on hte -log10(P-value), and colors are based on the generic label names.

        Parameters
        ----------
        self : Object
            The output of .fit_transform()
        node_size_limits : tuple
            node sizes are scaled between [min,max] values. The default is [6,15].
        savepath : str
            Save the figure in specified path.
        node_color : None or 'cluster' default : None
            color nodes based on clustering or by label colors.
        directed : bool, optional
            Create network using directed edges (arrows). The default is True.
        showfig : bool, optional
            Plot figure to screen. The default is True.

        Returns
        -------
        dict : containing various results derived from network.
        G : graph
            Graph generated by networkx.
        savepath : str
            Save the figure in specified path.
        labx : str
            labels of the nodes.

        """
        # Setup tempdir
        savepath = _tempdir(savepath)

        [IA,IB]=ismember(self.simmatLogP.columns, self.counts[:,0])
        node_size = np.repeat(node_size_limits[0], len(self.simmatLogP.columns))
        node_size[IA]=_scale_weights(self.counts[IB,1], node_size_limits)

        # Make undirected network
        simmatLogP = to_undirected(self.simmatLogP, method='logp')
        # Color node using network-clustering
        if node_color=='cluster':
            labx = self.plot(node_color='cluster', showfig=False)['labx']
        else:
            labx = label_encoder.fit_transform(self.labx)

        # Make network
        Gout = d3graphs(self.simmatLogP.T, savepath=savepath, node_size=node_size, charge=500, width=1500, height=800, collision=0.1, node_color=labx, directed=directed, showfig=showfig)
        # Return
        Gout['labx']=labx
        return(Gout)

    # Make network plot
    def plot(self, scale=2, dist_between_nodes=0.4, node_size_limits=[25,500], node_color=None, savepath=None, figsize=[15,10], pos=None, layout='fruchterman_reingold', dpi=250, showfig=True):
        """Make plot static network plot of the model results.

        Description
        -----------
        The results of hnet can be vizualized in several manners, one of them is a static network plot.

        Parameters
        ----------
        self : Object
            The output of .fit_transform()
        scale : int, optional
            scale the network by blowing it up by scale. The default is 2.
        dist_between_nodes : float, optional
            Distance between the nodes. Edges are sized based this value. The default is 0.4.
        node_size_limits : int, optional
            Nodes are scaled between the Min and max size. The default is [25,500].
        node_color : str, None or 'cluster' default is None
            color nodes based on clustering or by label colors.
        savepath : str, optional
            Save the figure in specified path.
        figsize : tuple, optional
            Size of the figure, [height,width]. The default is [15,10].
        pos : list, optional
            list with coordinates to orientate the nodes.
        layout : str, optional
            layouts from networkx can be used. The default is 'fruchterman_reingold'.
        dpi : int, optional
            resolution of the figure. The default is 250.
        showfig : bool, optional
            Plot figure to screen. The default is True.

        Returns
        -------
        Dict.
            Dictionary containing various results derived from network. The keys in the dict contain the following results:

                G : graph
            Graph generated by networkx.
        labx : str
            labels of the nodes.
        pos :  list
            Coordinates of the node postions.

        """
        config=dict()
        config['scale']=scale
        config['node_color']=node_color
        config['dist_between_nodes']=dist_between_nodes  # k=0.4
        config['node_size_limits']=node_size_limits
        config['layout']=layout
        config['iterations']=50
        config['dpi']=dpi

        # Set savepath and filename
        config['savepath']=_path_correct(savepath, filename='hnet_network', ext='.png')

        # Get adjmat
        adjmatLog = self.simmatLogP.copy()

        # Set weights for edges
        adjmatLogWEIGHT = adjmatLog.copy()
        np.fill_diagonal(adjmatLogWEIGHT.values, 0)
        adjmatLogWEIGHT = pd.DataFrame(index=adjmatLog.index.values, data=MinMaxScaler(feature_range=(0, 20)).fit_transform(adjmatLogWEIGHT), columns=adjmatLog.columns)

        # Set size for node
        [IA, IB]=ismember(self.simmatLogP.columns, self.counts[:, 0])
        node_size = np.repeat(node_size_limits[0], len(self.simmatLogP.columns))
        node_size[IA] = _scale_weights(self.counts[IB, 1], node_size_limits)

        # Make new graph (G) and add properties to nodes
        G = nx.DiGraph(directed=True)
        for i in range(0, adjmatLog.shape[0]):
            G.add_node(adjmatLog.index.values[i], node_size=node_size[i], node_label=self.labx[i])
        # Add properties to edges
        np.fill_diagonal(adjmatLog.values, 0)
        for i in range(0, adjmatLog.shape[0]):
            idx=np.where(adjmatLog.iloc[i,:]>0)[0]
            labels=adjmatLog.iloc[i, idx]
            labelsWEIGHT=adjmatLogWEIGHT.iloc[i,idx]

            for k in range(0, len(labels)):
                G.add_edge(labels.index[k], adjmatLog.index[i], weight=labels.values[k].astype(int), capacity=labelsWEIGHT.values[k])

        edges = G.edges()
        edge_weights = np.array([G[u][v]['weight'] for u,v in edges])
        edge_weights = MinMaxScaler(feature_range=(0.5, 8)).fit_transform(edge_weights.reshape(-1, 1)).flatten()

        # Cluster
        if node_color=='cluster':
            _, labx = network.cluster(G.to_undirected())
        else:
            labx = label_encoder.fit_transform(self.labx)

        # G = nx.DiGraph() # Directed graph
        # Layout
        if isinstance(pos, type(None)):
            pos = nx.fruchterman_reingold_layout(G, weight='edge_weights', k=config['dist_between_nodes'], scale=config['scale'], iterations=config['iterations'])
        else:
            pos = network.graphlayout(G, pos=pos, scale=config['scale'], layout=config['layout'])

        # pos = nx.spring_layout(G, weight='edge_weights', k=config['dist_between_nodes'], scale=config['scale'], iterations=config['iterations'])
        # Boot-up figure
        if showfig or (not isinstance(config['savepath'], type(None))):
            [fig, ax]=plt.subplots(figsize=figsize)
            options = {
                # 'node_color': 'grey',
                'arrowsize': 12,
                'font_size':18,
                'font_color':'black'}
            # Draw plot
            nx.draw(G, pos, with_labels=True, **options, node_size=node_size * 5, width=edge_weights, node_color=labx, cmap='Paired')
            # Plot weights
            nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))
            # Figure cleaning
            ax = plt.gca()
            ax.set_axis_off()
            # Show figure
            if showfig: plt.show()
            # Save figure to disk
            if not isinstance(config['savepath'], type(None)):
                savefig(fig, config['savepath'], dpi=config['dpi'], transp=True)

        # Return
        Gout = {}
        Gout['G']=G
        Gout['labx']=labx
        Gout['pos']=pos
        return(Gout)

    # Make plot of the structure_learning
    def heatmap(self, cluster=False, figsize=[15,10], savepath=None, verbose=3):
        """Plot heatmap.

        Description
        -----------
        A heatmap can be of use when the results becomes too large to plot in a network.

        Parameters
        ----------
        self : Object
            The output of .fit_transform()
        cluster : Bool, optional
            Cluster before making heatmap. The default is False.
        figsize : typle, optional
            Figure size. The default is [15, 10].
        savepath : Bool, optional
            saveingpath. The default is None.
        verbose : int, optional
            Verbosity. The default is 3.

        Returns
        -------
        None.

        """
        adjmatLog = self.simmatLogP.copy()
        # Set savepath and filename
        savepath = _path_correct(savepath, filename='hnet_heatmap', ext='.png')
    
        try:
            savepath1=''
            if savepath is not None:
                [getdir, getname, getext]=_path_split(savepath)
                if getname=='': getname='heatmap'
                savepath1 = os.path.join(getdir, getname + '_logP' + getext)
    
            if cluster:
                fig1=imagesc.cluster(adjmatLog.fillna(value=0).values, row_labels=adjmatLog.index.values, col_labels=adjmatLog.columns.values, cmap='Reds', figsize=figsize)
            else:
                fig1=imagesc.plot(adjmatLog.fillna(value=0).values, row_labels=adjmatLog.index.values, col_labels=adjmatLog.columns.values, cmap='Reds', figsize=figsize)
    
            if savepath is not None:
                if verbose>=3: print('[HNET.heatmap] Saving figure..')
                _ = savefig(fig1, savepath1, transp=True)
        except:
            print('[HNET.heatmap] Error: Heatmap failed. Try cluster=False')

    # Extract combined rules from structure_learning
    def combined_rules(self, verbose=3):
        """Association testing and combining Pvalues using fishers-method.
    
        Description
        -----------
        Multiple variables (antecedents) can be associated to a single variable (consequent).
        To test the significance of combined associations we used fishers-method. The strongest connection will be sorted on top.
    
        Parameters
        ----------
        model : dict
            The output of .fit()
        verbose : int, optional
            Print message to screen. The higher the number, the more details. The default is 3.
    
        Returns
        -------
        pd.DataFrame()
            Dataset containing antecedents and consequents. The strongest connection will be sorted on top.
            The columns are as following:
        antecedents_labx
            Generic label name.
        antecedents
            Specific label names in the 'from' catagory.
        consequents
            Specific label names that are the result of the antecedents.
        Pfisher
            Combined P-value
    
        Examples
        --------
        >>> hn = hnet()
        >>> df = hn.import_example('sprinkler')
        >>> hn.fit_transform(df)
        >>> hn.combined_rules()
        >>> print(hn.rules)

        """
        if not hasattr(self, 'simmatP'):
            raise Exception('[HNET.combined_rules] Error: Input requires the result from the hnet.fit() function.')

        df_rules = pd.DataFrame(index=np.arange(0, self.simmatP.shape[0]), columns=['antecedents_labx', 'antecedents', 'consequents', 'Pfisher'])
        df_rules['consequents'] = self.simmatP.index.values

        for i in tqdm(range(0, self.simmatP.shape[0]), disable=(True if verbose==0 else False)):
            idx=np.where(self.simmatP.iloc[i, :]<1)[0]
            # Remove self
            idx=np.setdiff1d(idx, i)
            # Store rules
            df_rules['antecedents'].iloc[i] = list(self.simmatP.iloc[i, idx].index)
            df_rules['antecedents_labx'].iloc[i] = self.labx[idx]
            # Combine pvalues
            df_rules['Pfisher'].iloc[i] = combine_pvalues(self.simmatP.iloc[i, idx].values, method='fisher')[1]

        # Keep only lines with pvalues
        df_rules.dropna(how='any', subset=['Pfisher'], inplace=True)
        # Sort
        df_rules.sort_values(by=['Pfisher'], ascending=True, inplace=True)
        df_rules.reset_index(inplace=True, drop=True)
        # Return
        return(df_rules)

    def import_example(data='titanic', verbose=3):
        return import_example(data=data, verbose=verbose)

# %% Import example dataset from github.
def import_example(data='titanic', verbose=3):
    """Import example dataset from github source.

    Parameters
    ----------
    data : str, optional
        Name of the dataset 'sprinkler' or 'titanic' or 'student'.
    verbose : int, optional
        Print message to screen. The default is 3.

    Returns
    -------
    pd.DataFrame()
        Dataset containing mixed features.

    """
    if data=='sprinkler':
        url='https://erdogant.github.io/datasets/sprinkler.zip'
    elif data=='titanic':
        url='https://erdogant.github.io/datasets/titanic_train.zip'
    elif data=='student':
        url='https://erdogant.github.io/datasets/student_train.zip'

    curpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    PATH_TO_DATA = os.path.join(curpath, wget.filename_from_url(url))

    # Check file exists.
    if not os.path.isfile(PATH_TO_DATA):
        if verbose>=3: print('[hnet] Downloading example dataset from github source..')
        wget.download(url, curpath)

    # Import local dataset
    if verbose>=3: print('[hnet] Import dataset [%s]' %(data))
    df = pd.read_csv(PATH_TO_DATA)
    # Return
    return df

# %% Compute fit
def enrichment(df, y, y_min=None, alpha=0.05, multtest='holm', dtypes='pandas', specificity='medium', verbose=3):
    """Enrichment analysis.

    Description
    -----------
    Compute enrichment between input dataset and response variable y. Length of dataframe and y must be equal.
    The input dataset is converted into a one-hot dense array based on automatic typing ``dtypes='pandas'`` or user defined dtypes.

    Parameters
    ----------
    df : DataFrame
        Input Dataframe.
    y : list of length df.index
        Response variable.
    y_min : int, optional
        Minimal number of samples in a group.. The default is None.
    alpha : float, optional
        Significance. The default is 0.05.
    multtest : String, optional
        Multiple test correcton. The default is 'holm'.
    dtypes : list of length df.columns, optional
        By default the dtype is determined based on the pandas dataframe. Empty ones [''] are skipped. The default is 'pandas'.
    specificity : String, optional
        Configure how numerical data labels are stored.. The default is 'medium'.
    verbose : int, optional
        Print message to screen. The higher the number, the more details. The default is 3.

    Returns
    -------
    pd.DataFrame() with the following columns:

    category_label : str
        Label of the catagory.
    P :  float
        Pvalue of the hypergeometric test or Wilcoxon Ranksum.
    logP :  float
        -log10(Pvalue) of the hypergeometric test or Wilcoxon Ranksum.
    Padj :  float
        Adjusted P-value.
    dtype : list of str
        Categoric or numeric.
    y : str
        Response variable name.
    category_name : str
        Subname of the category_label.
    popsize_M : int
        Population size: Total number of samples.
    nr_succes_pop_n : int
        Number of successes in population.
    overlap_X : int
        Overlap between response variable y and input feature.
    samplesize_N : int
        Sample size: Random variate, eg clustersize or groupsize, those of interest.
    zscore : float
        Z-score of the Wilcoxon Ranksum test.
    nr_not_succes_pop_n : int
        Number of successes in population.

    Examples
    --------
    >>> df = hnet.import_example('titanic')
    >>> y = df['Survived'].values
    >>> out = hnet.enrichment(df, y)

    """
    assert isinstance(df, pd.DataFrame), 'Data must be of type pd.DataFrame()'
    assert len(y)==df.shape[0], 'Length of [df] and [y] must be equal'
    assert 'numpy' in str(type(y)), 'y must be of type numpy array'

    # DECLARATIONS
    config = dict()
    config['verbose'] = verbose
    config['alpha'] = alpha
    config['multtest'] = multtest
    config['specificity'] = specificity

    if config['verbose']>=3: print('[HNET] Start making fit..')
    df.columns = df.columns.astype(str)
    # Set y as string
    y = df2onehot.set_y(y, y_min=y_min, verbose=config['verbose'])
    # Determine dtypes for columns
    [df, dtypes] = df2onehot.set_dtypes(df, dtypes, verbose=config['verbose'])
    # Compute fit
    out = _compute_significance(df, y, dtypes, specificity=config['specificity'], verbose=config['verbose'])
    # Multiple test correction
    out = _multipletestcorrection(out, config['multtest'], verbose=config['verbose'])
    # Keep only significant ones
    out = _filter_significance(out, config['alpha'], multtest)
    # Make dataframe
    out = pd.DataFrame(out)
    # Return
    if config['verbose']>=3: print('[HNET] Fin')
    return(out)


# %% Compute significance
def _compute_significance(df, y, dtypes, specificity=None, verbose=3):
    out=[]
    # Run over all columns
    for i in range(0, df.shape[1]):
        if (i>0) and (verbose>=3): print('')
        if verbose>=3: print('[HNET] Analyzing [%s] %s' %(dtypes[i], df.columns[i]), end='')
        colname = df.columns[i]

        # Clean nan fields
        [datac, yc] = _nancleaning(df[colname], y)
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
    simmat_padj[simmat_padj==0]=1e-323
    adjmatLog=(-np.log10(simmat_padj)).copy()
    adjmatLog[adjmatLog == -np.inf] = np.nanmax(adjmatLog[adjmatLog != np.inf])
    adjmatLog[adjmatLog == np.inf] = np.nanmax(adjmatLog[adjmatLog != np.inf])
    adjmatLog[adjmatLog == -0] = 0
    return(adjmatLog)


# %% Add columns
def _addcolumns(simmat_padj, colnames, Xlabx, catnames):
    I=np.isin(colnames.values.astype(str), simmat_padj.index.values)
    if np.any(I):
        newcols=list((colnames.values[I == False]).astype(str))
        newcats=list((catnames[I == False]).astype(str))

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
            if verbose>=3: print('[HNET] %.0f columns are removed.' %(len(remcols)))

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
                if verbose>=3: print('[HNET] Dropping %s' %(col))
                droplabel.append(col)

    # Remove labels from the original df
    Xlabx=Xlabx[np.isin(dfO.columns, droplabel)==False]
    dfO.drop(labels=droplabel, axis=0, inplace=True)
    dfO.drop(labels=droplabel, axis=1, inplace=True)

    return(dfO, Xlabx)


# %% Do multiple test correction
def _multipletestcorrectionAdjmat(adjmat, multtest, verbose=3):
    if verbose>=3: print('[HNET] Multiple test correction using %s' %(multtest))
    # Multiple test correction
    if not (isinstance(multtest, type(None))):
        # Make big row with all pvalues
        tmpP=adjmat.values.ravel()
        # Find not nans
        Iloc = ~np.isnan(tmpP)
        Padj=np.zeros_like(tmpP) * np.nan
        # Do multiple test correction on only the tested ones
        Padj[Iloc]=multitest.multipletests(tmpP[Iloc], method=multtest)[1]
        # Rebuild adjmatrix
        adjmat = pd.DataFrame(data=Padj.reshape(adjmat.shape), columns=adjmat.columns, index=adjmat.index)

    # Return
    return(adjmat)


# %% Do multiple test correction
def _multipletestcorrection(out, multtest, verbose=3):
    # Always do a multiple test correction but do not use it in the filtering step if not desired
    if verbose>=3: print('[HNET] Multiple test correction using %s' %(multtest))

    if out!=[]:
        # Get pvalues
        Praw = np.array(list(map(lambda x: x['P'], out)))
        Iloc = np.isnan(Praw)
        Praw[Iloc] = 1

        # Multiple test correction
        if (isinstance(multtest, type(None))):
            Padj=Praw
        else:
            # Padj=np.zeros_like(Praw)*np.nan
            Padj=multitest.multipletests(Praw, method=multtest)[1]

        for i in range(0, len(out)):
            out[i].update({'Padj':Padj[i]})

    return(out)


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
                if verbose>=3: print('[HNET] Adding %d none mutual exclusive combinations with k=[%d] features.' %(cmbn_hot.shape[1], k))
                out_hot = pd.concat([out_hot, pd.DataFrame(data=cmbn_hot, columns=cmbn_labH).astype(int)], axis=1)
                out_labo = np.append(out_labo, cmbn_labO, axis=0)
                out_labx = out_labx + cmbn_labX
            else:
                if verbose>=3: print('[HNET] No combinatorial features detected with k=[%d] features. No need to search for higher k.' %(k))
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


# %% Do the math
def _do_the_math(df, X_comb, dtypes, X_labx, simmat_padj, simmat_labx, i, specificity, y_min, verbose=3):
    count=0
    out=[]
    # Get response variable to test association
    y=X_comb.iloc[:,i].values.astype(str)
    # Get column name
    colname=X_comb.columns[i]
    # Do something if response variable has more then 1 option
    if len(np.unique(y))>1:
        if verbose>=4: print('[HNET] Working on [%s]' %(X_comb.columns[i]), end='')
        # Remove columns if it belongs to the same categorical subgroup; these can never overlap!
        Iloc = ~np.isin(df.columns, X_labx[i])
        # Compute fit
        dfout = enrichment(df.loc[:,Iloc], y, y_min=y_min, alpha=1, multtest=None, dtypes=dtypes[Iloc], specificity=specificity, verbose=0)
        # Count
        count=count + dfout.shape[0]
        # Match with dataframe and store
        if not dfout.empty:
            # Column names
            idx = np.where(dfout['category_label'].isna())[0]
            catnames = dfout['category_name']
            colnames = catnames + '_' + dfout['category_label']
            colnames[idx] = catnames[idx].values
            # Add new column and index
            [simmat_padj, simmat_labx]=_addcolumns(simmat_padj, colnames, simmat_labx, catnames)
            # Store values
            [IA,IB]=ismember(simmat_padj.index.values.astype(str), colnames.values.astype(str))
            simmat_padj.loc[colname, IA] = dfout['Padj'].iloc[IB].values
            # Count nr. successes
            out = [colname, X_comb.iloc[:,i].sum() / X_comb.shape[0]]
            # showprogress
            if verbose>=4: print('[%g]' %(len(IB)), end='')
    else:
        if verbose>=4: print('[HNET] Skipping [%s] because length of unique values=1' %(X_comb.columns[i]), end='')

    if verbose>=4: print('')
    # Return
    return(out, simmat_padj, simmat_labx)


# %% Do the math
def _post_processing(simmat_padj, nr_succes_pop_n, simmat_labx, alpha, multtest, fillna, dropna, verbose=3):
    # Clean label names by chaning X.0 into X
    simmat_padj.columns=list(map(lambda x: x[:-2] if x[-2:]=='.0' else x, simmat_padj.columns))
    simmat_padj.index=list(map(lambda x: x[:-2] if x[-2:]=='.0' else x, simmat_padj.index.values))
    nr_succes_pop_n=np.array(nr_succes_pop_n)
    nr_succes_pop_n[:,0]=list(map(lambda x: x[:-2] if x[-2:]=='.0' else x, nr_succes_pop_n[:,0]))

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

    return(simmat_padj, nr_succes_pop_n, adjmatLog, simmat_labx)


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


# %% Preprocessing
def _preprocessing(df, dtypes='pandas', y_min=10, perc_min_num=0.8, excl_background=None, verbose=3):
    df.columns = df.columns.astype(str)
    # Remove columns without dtype
    [df, dtypes] = _remove_columns_without_dtype(df, dtypes, verbose=verbose)
    # Make onehot matrix for response variable y
    df_onehot = df2onehot.df2onehot(df, dtypes=dtypes, y_min=y_min, hot_only=True, perc_min_num=perc_min_num, excl_background=excl_background, verbose=verbose)
    dtypes = df_onehot['dtypes']
    # Some check before proceeding
    assert (not df_onehot['onehot'].empty) or (not np.all(np.isin(dtypes, 'num'))), '[HNET] ALL data is excluded from the dataframe! There should be at least 1 categorical value!'
    assert df.shape[1] == len(dtypes), '[HNET] DataFrame Shape and dtypes length does not match'
    # Make all integer
    df_onehot['onehot'] = df_onehot['onehot'].astype(int)
    # Return
    return df, df_onehot, dtypes


# %% Tempdir
def _tempdir(savepath):
    if savepath is None:
        savepath = os.path.join(tempfile.gettempdir(), '')
    return(savepath)


# %% Make adjacency matrix symmetric with repect to the diagonal
def to_undirected(adjmat, method='logp', verbose=3):
    """Make adjacency matrix symmetric.

    Description
    -----------
    The adjacency matrix resulting from hnet is not neccesarily symmetric due to the statistics being used.
    In some cases, a symmetric matrix can be usefull. This function makes sure that values above the diagonal are the same as below the diagonal.
    Values above and below the diagnal are combined using the max or min value.

    Parameters
    ----------
    adjmat : array
        Square form adjacency matrix.
    method : str
        Make matrix symmetric using the 'max' or 'min' function.
    verbose : int
        Verbosity. The default is 3.

    Returns
    -------
    pd.DataFrame().
        Symmetric adjacency matrix.

    """
    if np.sum(np.isin(adjmat.index.values, adjmat.columns.values))!=np.max(adjmat.shape):
        raise Exception('Adjacency matrix must have similar number of rows and columns! Re-run HNet with dropna=False!')

    progressbar=(True if verbose==0 else False)
    columns=adjmat.columns.values
    index=adjmat.index.values

    # Make selected matrix symmetric
    if isinstance(adjmat, pd.DataFrame):
        adjmat=adjmat.values

    # Make output matrix
    adjmatS=np.zeros(adjmat.shape, dtype=float)

    # Make symmetric using maximum as combining function
    for i in tqdm(range(adjmat.shape[0]), disable=progressbar):
        for j in range(adjmat.shape[1]):
            if method=='logp':
                # Maximum -log10(P) values
                score = np.maximum(adjmat[i, j], adjmat[j, i])
            else:
                # Minimum P-values
                score = np.minimum(adjmat[i, j], adjmat[j, i])

            adjmatS[i, j] = score
            adjmatS[j, i] = score

    # Make dataframe and return
    adjmatS=pd.DataFrame(index=index, data=adjmatS, columns=columns, dtype=float)
    # Return
    return(adjmatS)


# %% Comparison of two networks
def compare_networks(adjmat_true, adjmat_pred, pos=None, showfig=True, width=15, height=8, verbose=3):
    """Compare two adjacency matrices and plot the differences.

    Description
    -----------
    Comparison of two networks based on two adjacency matrices. Both matrices should be of equal size and of type pandas DataFrame.
    The columns and rows between both matrices are matched if not ordered similarly.

    Parameters
    ----------
    adjmat_true : pd.DataFrame()
        First array.
    adjmat_pred : pd.DataFrame()
        Second array.
    pos : dict, optional
        Position of the nodes. The default is None.
    showfig : Bool, optional
        Plot figure. The default is True.
    width : int, optional
        Width of the figure. The default is 15.
    height : int, optional
        Height of the figure. The default is 8.
    verbose : int, optional
        Verbosity. The default is 3.

    Returns
    -------
    tuple
        Output contains a tuple of two elements, the score of matching adjacency matrix and adjacency matrix differences.
    scores : dict
        Contains extensive number of keys with various scoring values.
    adjmat_diff : pd.DataFrame()
        Differences between the two output matrices. Zero means no difference whereas value >0 does.

    """
    [scores, adjmat_diff] = network.compare_networks(adjmat_true,
                                                     adjmat_pred,
                                                     pos=pos,
                                                     showfig=showfig,
                                                     width=width,
                                                     height=height,
                                                     verbose=verbose)
    return(scores, adjmat_diff)


# %% Main
if __name__ == "__main__":
    import hnet as hnet
    df = hnet.import_example('titanic')
    out = hnet.fit(df)
    G = hnet.d3graph(out)
