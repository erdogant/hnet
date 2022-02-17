"""HNET: Graphical Hypergeometric-networks."""
# -------------------------------------------------
# Name        : hnet.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# github      : https://github.com/erdogant/hnet
# Licence     : See licences
# -------------------------------------------------

# %% Libraries
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import wget
import networkx as nx
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from scipy.stats import combine_pvalues
import hnet.hnstats as hnstats
import hnet.network as network
from hnet.savefig import savefig
import pypickle
import colourmap
import df2onehot
import imagesc
from ismember import ismember
from d3heatmap import d3heatmap as d3
from d3graph import d3graph as d3graphs
import warnings
warnings.filterwarnings("ignore")

# Custom packages

# Local utils

# Known libraries
label_encoder = LabelEncoder()


# Internal

# from functools import lru_cache

# %% Association learning across all variables

class hnet():
    """HNET - Graphical Hypergeometric networks.

    Description
    -----------
    This is the main function to detect significant edge probabilities between pairs of vertices (node-links) given the input DataFrame.

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
    alpha : float [0..1], (default : 0.05)
        Significance to keep only edges with <=alhpa.
        1 : (for all results)
    y_min : int [1..n], where n is the number of samples. (default : 10)
        Minimum number of samples in a group. Should be [samples>=y_min]. All groups with less then y_min samples are labeled as _other_ and are not used in the model.
        10, None, 1, etc
    perc_min_num : float [None, 0..1], optional
        Force column (int or float) to be numerical if unique non-zero values are above percentage.
    k : int, [1..n] , (default : 1)
        Number of combinatoric elements to create for the n features
    multtest : String, (default : 'holm')
        * None: No multiple Test,
        * 'bonferroni': one-step correction,
        * 'sidak': one-step correction,
        * 'holm-sidak': step down method using Sidak adjustments,
        * 'holm': step-down method using Bonferroni adjustments,
        * 'simes-hochberg': step-up method  (independent),
        * 'hommel': closed method based on Simes tests (non-negative),
        * 'fdr_bh': Benjamini/Hochberg  (non-negative),
        * 'fdr_by': Benjamini/Yekutieli (negative),
        * 'fdr_tsbh': two stage fdr correction (non-negative),
        * 'fdr_tsbky': two stage fdr correction (non-negative)
    dtypes : list of str, (default : 'pandas')
        list strings, example: ['cat','num',''] of length y. By default the dtype is determined based on the pandas dataframe. Empty ones [''] are skipped.
        Can also be of the form: ['cat','cat','num','','cat']
    specificity : String, (default : 'medium')
        Configure how numerical data labels are stored. Setting this variable can be of use in the 'association_learning' function for the creation of a network ([None] will glue most numerical labels together whereas [high] mostly will not).
        * None : No additional information in the labels,
        * 'low' : 'high' or 'low' are included that represents significantly higher or lower assocations compared to the rest-group,
        * 'medium': 'high' or 'low' are included with 1 decimal behind the comma,
        * 'high' : 'high' or 'low' are included with 3 decimal behind the comma.
    dropna : Bool, [True,False] (Default : True)
        Drop rows/columns in adjacency matrix that showed no significance
    excl_background : list or None, [0], [0, '0.0', 'male', ...],  (Default: None)
        Remove values/strings that labeled as background. As an example, in a two-class approach with [0,1], the 0 is usually the background and not of interest.
        Example: '0.0': To remove categorical values with label 0
    black_list : List or None (default : None)
        If a list of edges is provided as black_list, they are excluded from the search and the resulting model will not contain any of those edges.
    white_list : List or None (default : None)
        If a list of edges is provided as white_list, the search is limited to those edges. The resulting model will then only contain edges that are in white_list.

    Returns
    -------
    dict()
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
    >>> from hnet import hnet
    >>> hn = hnet()
    >>> # Load example dataset
    >>> df = hn.import_example('sprinkler')
    >>> association Learning
    >>> out = hn.association_learning(df)
    >>> # Plot dynamic graph
    >>> G_dynamic = hn.d3graph()
    >>> # Plot static graph
    >>> G_static = hn.plot()
    >>> # Plot heatmap
    >>> P_heatmap = hn.heatmap(cluster=True)
    >>> # Plot feature importance
    >>> hn.plot_feat_importance()
    """

    def __init__(self, alpha=0.05, y_min=10, perc_min_num=0.8, k=1, multtest='holm', dtypes='pandas', specificity='medium', dropna=True, excl_background=None, black_list=None, white_list=None):
        """Initialize distfit with user-defined parameters."""
        if (alpha is None): alpha=1
        if (y_min is None): y_min=1
        if isinstance(white_list, str): white_list=[white_list]
        if isinstance(black_list, str): black_list=[black_list]
        if (white_list is not None) and len(white_list)==0: white_list=None
        if (black_list is not None) and len(black_list)==0: black_list=None
        # Store in object
        self.alpha = alpha
        self.y_min = np.maximum(1, y_min)
        self.k = k
        self.multtest = multtest
        self.dtypes = dtypes
        self.specificity = specificity
        self.perc_min_num = perc_min_num
        self.dropna = dropna
        self.fillna = True
        self.excl_background = excl_background
        self.white_list = white_list
        self.black_list = black_list

    def prepocessing(self, df, verbose=3):
        """Pre-processing based on the model parameters."""
        # Pre processing
        [df, df_onehot, dtypes] = hnstats._preprocessing(df, dtypes=self.dtypes, y_min=self.y_min, perc_min_num=self.perc_min_num, excl_background=self.excl_background, white_list=self.white_list, black_list=self.black_list, verbose=verbose)
        # Add combinations
        [X_comb, X_labx, X_labo] = hnstats._make_n_combinations(df_onehot['onehot'], df_onehot['labx'], self.k, self.y_min, verbose=verbose)
        # Get numerical columns
        colNum = df.columns[df.dtypes == 'float64'].values
        simmat_labx = np.append(X_labo, colNum).astype(str)
        simmatP = pd.DataFrame(index=np.append(X_comb.columns, colNum).astype(str), columns=np.append(X_comb.columns, colNum).astype(str)).astype(float)
        # Return
        return df, simmatP, simmat_labx, X_comb, X_labx, dtypes

    def compute_associations(self, df, simmatP, simmat_labx, X_comb, X_labx, dtypes, verbose=3):
        """Association learning on the processed data."""
        # Here we go! in parallel!
        # from multiprocessing import Pool
        # nr_succes_pop_n=[]
        # with Pool(processes=os.cpu_count()-1) as pool:
        #     for i in range(0,X_comb.shape[1]):
        #         result = pool.apply_async(_do_the_math, (df, X_comb, dtypes, X_labx, param, i,))
        #         nr_succes_pop_n.append(result)
        #     results = [result.get() for result in result_objs]
        #     print(len(results))

        # Print some
        if verbose>=3: print('[hnet] >Association learning across [%d] categories.' %(X_comb.shape[1]))

        disable = (True if (verbose==0 or verbose>3) else False)
        nr_succes_pop_n = []

        for i in tqdm(range(0, X_comb.shape[1]), disable=disable):
            nr_succes_i, simmatP, simmat_labx = _do_the_math(df, X_comb, dtypes, X_labx, simmatP, simmat_labx, i, self.specificity, self.y_min, verbose=verbose)
            nr_succes_pop_n.append(nr_succes_i)
            if verbose>=5: print('[hnet] >[%d] %s' %(i, nr_succes_i))

        # Post processing
        simmatP, simmatLogP, simmat_labx, nr_succes_pop_n = hnstats._post_processing(simmatP, nr_succes_pop_n, simmat_labx, self.alpha, self.multtest, self.fillna, self.dropna, verbose=verbose)
        # Message
        print_text = '[hnet] >Total number of associatons computed: [%.0d]' %((simmatP.shape[0] * simmatP.shape[1]) - simmatP.shape[0])
        if verbose>=3: print('-' * len(print_text))
        if verbose>=3: print(print_text)
        if verbose>=3: print('-' * len(print_text))
        # Return
        return simmatP, simmatLogP, simmat_labx, nr_succes_pop_n

    def association_learning(self, df, verbose=3):
        """Learn the associations in the data.

        Parameters
        ----------
        df : DataFrame, [NxM].
            N=rows->samples, and  M=columns->features.

           |    | f1| f2| f3|
           | s1 | 0 | 0 | 1 |
           | s2 | 0 | 1 | 0 |
           | s3 | 1 | 1 | 0 |

        verbose : int [1-5], default: 3
            Print information to screen. 0: nothing, 1: Error, 2: Warning, 3: information, 4: debug, 5: trace.

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
        if not isinstance(df, pd.DataFrame): raise ValueError('Input data [df] must be of type pd.DataFrame()')
        # Clean readily fitted models to ensure correct results.
        self._clean(verbose=verbose)
        # Preprocessing
        df, simmatP, labx, X_comb, X_labx, dtypes = self.prepocessing(df, verbose=verbose)
        # Here we go! Over all columns now
        simmatP, simmatLogP, labx, nr_succes_pop_n = self.compute_associations(df, simmatP, labx, X_comb, X_labx, dtypes, verbose=verbose)
        # Combine rules
        rules = self.combined_rules(simmatP, labx, verbose=0)
        # Feature importance
        feat_importance = self._feat_importance(simmatLogP, labx, verbose=verbose)
        # Compute associations for the catagories
        GsimmatP, GsimmatLogP = self._compute_associations_cat(simmatP, labx, verbose=verbose)
        # Store
        self.results = _store(simmatP, simmatLogP, GsimmatP, GsimmatLogP, labx, df, nr_succes_pop_n, dtypes, rules, feat_importance)
        # Use this option for storage of your model
        if verbose>=3: print('[hnet] >Fin.')
        return self.results

    def _compute_associations_cat(self, simmatP, labx, verbose=3):
        """Compute category associations by fishers method."""
        # Make some checks
        if not np.all(simmatP.columns==simmatP.index.values):
            raise ValueError('[hnet] >Error: Adjacency matrix [simmatP] does not have the same number of columns and index!')
        if not np.all(simmatP.shape[0]==len(labx)):
            raise ValueError('[hnet] >Error: The number of columns in [simmatP] does not match with the label [labx]!')

        if verbose>=3: print('[hnet] >Computing category association using fishers method..')
        uilabx = np.unique(labx)
        adjmatP = np.ones((len(uilabx), len(uilabx)))

        # Combine Pvalues for catagories based on fishers method
        for labxC in tqdm(uilabx, disable=(True if verbose==0 else False)):
            for labxR in uilabx:
                Ic = labx==labxC
                Ir = labx==labxR
                adjmatP[uilabx==labxR, uilabx==labxC] = combine_pvalues(simmatP.loc[Ir, Ic].values.ravel(), method='fisher')[1]

        # Store in dataframe
        adjmatP = pd.DataFrame(data=adjmatP, index=uilabx, columns=uilabx)
        adjmatlogP = -np.log10(adjmatP)
        adjmatlogP[adjmatlogP<=0]=0
        # Replace inf values with maximum value that is possible
        # adjmatlogP[np.isinf(adjmatlogP)]=323
        for (columnName, columnData) in adjmatlogP.iteritems():
            columnData.loc[~(columnData!= np.inf)] = columnData.loc[columnData!= np.inf].max()
        return adjmatP, adjmatlogP

    def _feat_importance(self, simmatLogP, labx, verbose=3):
        """Compute feature importance."""
        # Get unique labels
        uilabx, counts = np.unique(labx, return_counts=True)
        # Count the number of significant associations per category
        Pcounts = []
        Psum = []
        for lab in uilabx:
            Iloc = labx==lab
            # Sum over significance per label
            Psum.append(simmatLogP.iloc[Iloc, :].sum(axis=1).sum())
            # Count number of significant category labels
            Pcounts.append((simmatLogP.iloc[Iloc, :]>0).sum(axis=1).sum())

        # Compute corrected Pvalue
        Pcorr = (np.array(Psum) / np.array(Pcounts))
        Pcorr[np.isnan(Pcorr)]=0
        # Store
        df_feat_importance = pd.DataFrame(data=np.c_[Pcounts, Psum, Pcorr], columns=['degree', 'Psum', 'Pcorr'])
        df_feat_importance.index = uilabx
        df_feat_importance.sort_values(by='Pcorr', ascending=False, inplace=True)
        # Return
        return df_feat_importance

    # Make network d3
    def plot_feat_importance(self, marker_size=5, top_n=10, figsize=(15, 8), verbose=3):
        """Plot feature importance.

        Parameters
        ----------
        marker_size : int, (default: 5)
            Marker size in the scatter plot.
        top_n : int, optional
            Top n features are labelled in the plot. The default is 10.
        figsize : tuple, optional
            Size of the figure in the browser, [height,width]. The default is [1500,1500].
        verbose : int, optional
            Verbosity. The default is 3.

        Returns
        -------
        None.

        """
        # Make plot counts significant associations per label
        df = pd.DataFrame((self.results['simmatLogP']>0).sum(axis=1), columns=['Psum'])
        df.reset_index(drop=False, inplace=True)
        c = colourmap.fromlist(self.results['labx'])[0]
        plt.figure(figsize=figsize)
        plt.scatter(np.arange(0, len(df.values)), df['Psum'].values, color=c, s=marker_size)
        plt.grid(True)
        plt.xlabel('label')
        plt.ylabel('Degree')
        plt.title('Number of significant associations per label (degree)')
        # Add the top n labels
        top_n = np.minimum(top_n, self.results['feat_importance'].shape[0])
        df = df.sort_values(by='Psum', ascending=False)[0:top_n]
        for i in range(0, top_n):
            plt.text(df.index[i], df['Psum'].iloc[i], df['index'].iloc[i])

        # Plot the cataogires
        plt.figure(figsize=figsize)
        self.results['feat_importance']['degree'].plot(kind='bar', figsize=(15, 8))
        plt.grid(True)
        plt.xlabel('Catagories')
        plt.ylabel('Total degree (significant associations)')
        plt.title('Total degree per category; the number of significant edges.')

        # Plot feature importance after normalization
        plt.figure(figsize=figsize)
        self.results['feat_importance']['Pcorr'].plot(kind='bar')
        plt.grid(True)
        plt.xlabel('Catagories')
        plt.ylabel('Normalized significance')
        plt.title('Feature importance after normalization. Per category: E(-log10(P) / E(significant edges per node)')

    # Get adjacency matrix
    def _get_adjmat(self, summarize):
        """Retrieve data for catagories or labels."""
        if summarize:
            simmatP = self.results['simmatP_cat']
            simmatLogP = self.results['simmatLogP_cat']
            labx = self.results['simmatP_cat'].columns.values
        else:
            simmatP = self.results['simmatP']
            simmatLogP = self.results['simmatLogP']
            labx = self.results['labx']
        return simmatP, simmatLogP, labx

    # Make network d3
    def d3heatmap(self, summarize=False, savepath=None, directed=True, threshold=None, white_list=None, black_list=None, min_edges=None, figsize=(700, 700), vmax=None, showfig=True, verbose=3):
        """Interactive heatmap creator.

        Description
        -----------
        This function creates a interactive and stand-alone heatmap that is build on d3 javascript.
        d3heatmap is integrated into hnet and uses the -log10(P-value) adjacency matrix.
        Each column and index name represents a node whereas values >0 in the matrix represents an edge.
        Node links are build from rows to columns. Building the edges from row to columns only matters in directed cases.
        The network nodes and edges are adjusted in weight based on hte -log10(P-value), and colors are based on the category names.

        Parameters
        ----------
        self : Object
            The output of .association_learning()
        summarize : bool, (default: False)
            Show the results based on categoric or label-specific associations.
            True: Summrize based on the categories
            False: All associations across labels
        savepath : str
            Save the figure in specified path.
        directed : bool, default is True.
            Create network using directed edges (arrows).
        threshold : int (default : None)
            Associations (edges) are filtered based on the -log10(P) > threshold. threshold should range between 0 and maximum value of -log10(P).
        black_list : List or None (default : None)
            If a list of edges is provided as black_list, they are excluded from the search and the resulting model will not contain any of those edges.
        white_list : List or None (default : None)
            If a list of edges is provided as white_list, the search is limited to those edges. The resulting model will then only contain edges that are in white_list.
        min_edges : int (default : None)
            Edges are only shown if a node has at least min_edges.
        showfig : bool, optional
            Plot figure to screen. The default is True.
        figsize : tuple, optional
            Size of the figure in the browser, [height,width]. The default is [1500,1500].

        Returns
        -------
        dict : containing various results derived from network.
        savepath : str
            Save the figure in specified path.
        labx : array-like
            Cluster labels.

        """
        if verbose>=3: print('[hnet] >Building dynamic heatmaps using d3graph..')
        # Check results
        status = self._check_results()
        if not status: return None
        # Retrieve data
        _, simmatLogP, labx = self._get_adjmat(summarize)

        # Filter adjacency matrix on blacklist/whitelist and/or threshold
        simmatLogP, labx = hnstats._filter_adjmat(simmatLogP, labx, threshold=threshold, min_edges=min_edges, white_list=white_list, black_list=black_list, verbose=verbose)
        # Check whether anything has remained
        if simmatLogP.values.flatten().sum()==0:
            if verbose>=3: print('[hnet] >Nothing to plot.')
            return None

        # Make undirected network
        if not directed:
            simmatLogP = to_undirected(simmatLogP, method='logp')

        # Cluster
        labx = self.plot(summarize=summarize, node_color='cluster', directed=True, threshold=threshold, white_list=white_list, black_list=black_list, min_edges=min_edges, showfig=False)['labx']

        if vmax is None:
            vmax = np.max(np.max(simmatLogP)) / 10

        # Make heatmap
        if verbose>=3: print('[hnet] >Creating output html..')
        paths = d3.heatmap(simmatLogP, clust=labx, path=savepath, title='Hnet d3heatmap', vmax=vmax, width=figsize[1], height=figsize[0], showfig=showfig, stroke='red', verbose=verbose)

        # Return
        results = {}
        results['paths'] = paths
        results['clust_labx'] = labx
        return(results)

    # Make network d3
    def d3graph(self, summarize=False, node_size_limits=[6, 15], savepath=None, node_color=None, directed=True, threshold=None, white_list=None, black_list=None, min_edges=None, charge=500, figsize=(1500, 1500), showfig=True, verbose=3):
        """Interactive network creator.

        Description
        -----------
        This function creates a interactive and stand-alone network that is build on d3 javascript.
        d3graph is integrated into hnet and uses the -log10(P-value) adjacency matrix.
        Each column and index name represents a node whereas values >0 in the matrix represents an edge.
        Node links are build from rows to columns. Building the edges from row to columns only matters in directed cases.
        The network nodes and edges are adjusted in weight based on hte -log10(P-value), and colors are based on the category names.

        Parameters
        ----------
        self : Object
            The output of .association_learning()
        summarize : bool, (default: False)
            Show the results based on categoric or label-specific associations.
            True: Summrize based on the categories
            False: All associations across labels
        node_size_limits : tuple
            node sizes are scaled between [min,max] values. The default is [6,15].
        savepath : str
            Save the figure in specified path.
        node_color : None or 'cluster' default : None
            color nodes based on clustering or by label colors.
        directed : bool, default is True.
            Create network using directed edges (arrows).
        threshold : int (default : None)
            Associations (edges) are filtered based on the -log10(P) > threshold. threshold should range between 0 and maximum value of -log10(P).
        black_list : List or None (default : None)
            If a list of edges is provided as black_list, they are excluded from the search and the resulting model will not contain any of those edges.
        white_list : List or None (default : None)
            If a list of edges is provided as white_list, the search is limited to those edges. The resulting model will then only contain edges that are in white_list.
        min_edges : int (default : None)
            Edges are only shown if a node has at least min_edges.
        showfig : bool, optional
            Plot figure to screen. The default is True.
        figsize : tuple, optional
            Size of the figure in the browser, [height,width]. The default is [1500,1500].

        Returns
        -------
        dict : containing various results derived from network.
        G : graph
            Graph generated by networkx.
        savepath : str
            Save the figure in specified path.
        labx : array-like
            Cluster labels.

        """
        if verbose>=3: print('[hnet] >Building dynamic network graph using d3graph..')
        # Check input data
        status = self._check_results()
        if not status: return None
        # Retrieve data
        _, simmatLogP, labx = self._get_adjmat(summarize)
        # Setup tempdir
        savepath = hnstats._tempdir(savepath)
        # Filter adjacency matrix on blacklist/whitelist and/or threshold
        simmatLogP, labx = hnstats._filter_adjmat(simmatLogP, labx, threshold=threshold, min_edges=min_edges, white_list=white_list, black_list=black_list, verbose=verbose)
        # Check whether anything has remained
        if simmatLogP.values.flatten().sum()==0:
            if verbose>=3: print('[hnet] >Nothing to plot.')
            return None

        # Resizing nodes based on user-limits
        IA, IB = ismember(simmatLogP.columns, self.results['counts'][:, 0])
        node_size = np.repeat(node_size_limits[0], len(simmatLogP.columns))
        node_size[IA] = hnstats._scale_weights(self.results['counts'][IB, 1], node_size_limits)

        # Make undirected network
        if not directed:
            simmatLogP = to_undirected(simmatLogP, method='logp')

        # Color node using network-clustering
        if node_color=='cluster':
            labx = self.plot(summarize=summarize, node_color='cluster', directed=True, threshold=threshold, white_list=white_list, black_list=black_list, min_edges=min_edges, showfig=False)['labx']
        else:
            labx = label_encoder.fit_transform(labx)

        # Make network
        if verbose>=3: print('[hnet] >Creating output html..')
        Gout = d3graphs(simmatLogP.T, savepath=savepath, node_size=node_size, charge=charge, height=figsize[0], width=figsize[1], collision=0.1, node_color=labx, directed=directed, showfig=showfig)
        # Return
        Gout['labx'] = labx
        return(Gout)

    # Make network plot
    def plot(self, summarize=False, scale=2, dist_between_nodes=0.4, node_size_limits=[25, 500], directed=True, node_color=None, savepath=None, figsize=[15, 10], pos=None, layout='fruchterman_reingold', dpi=250, threshold=None, white_list=None, black_list=None, min_edges=None, showfig=True, verbose=3):
        """Make plot static network plot of the model results.

        Description
        -----------
        The results of hnet can be vizualized in several manners, one of them is a static network plot.

        Parameters
        ----------
        self : Object
            The output of .association_learning()
        summarize : bool, (default: False)
            Show the results based on categoric or label-specific associations.
            True: Summrize based on the categories
            False: All associations across labels
        scale : int, optional
            scale the network by blowing it up by scale. The default is 2.
        dist_between_nodes : float, optional
            Distance between the nodes. Edges are sized based this value. The default is 0.4.
        node_size_limits : int, optional
            Nodes are scaled between the Min and max size. The default is [25,500].
        node_color : str, None or 'cluster' default is None
            color nodes based on clustering or by label colors.
        directed : bool, default is True.
            Create network using directed edges (arrows).
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
        threshold : int (default : None)
            Associations (edges) are filtered based on the -log10(P) > threshold. threshold should range between 0 and maximum value of -log10(P).
        black_list : List or None (default : None)
            If a list of edges is provided as black_list, they are excluded from the search and the resulting model will not contain any of those edges.
        white_list : List or None (default : None)
            If a list of edges is provided as white_list, the search is limited to those edges. The resulting model will then only contain edges that are in white_list.
        min_edges : int (default : None)
            Edges are only shown if a node has at least min_edges.
        showfig : bool, optional
            Plot figure to screen. The default is True.

        Returns
        -------
        dict. Dictionary containing various results derived from network. The keys in the dict contain the following results:
        G : graph
            Graph generated by networkx.
        labx : str
            labels of the nodes.
        pos :  list
            Coordinates of the node postions.

        """
        if verbose>=3: print('[hnet] >Building static network graph..')
        # Check input data
        status = self._check_results()
        if not status: return None
        # Retrieve data
        _, simmatLogP, labx = self._get_adjmat(summarize)

        config = {}
        config['scale'] = scale
        config['node_color'] = node_color
        config['dist_between_nodes'] = dist_between_nodes
        config['node_size_limits'] = node_size_limits
        config['layout'] = layout
        config['iterations'] = 50
        config['dpi'] = dpi

        # Set savepath and filename
        config['savepath'] = hnstats._path_correct(savepath, filename='hnet_network', ext='.png')
        # Get adjmat
        # adjmatLog = self.results['simmatLogP'].copy()
        # Filter adjacency matrix on blacklist/whitelist and/or threshold
        adjmatLog, labx = hnstats._filter_adjmat(simmatLogP, labx, threshold=threshold, min_edges=min_edges, white_list=white_list, black_list=black_list, verbose=verbose)
        # Check whether anything has remained
        if adjmatLog.values.flatten().sum()==0:
            if verbose>=3: print('[hnet] >Nothing to plot.')
            return None

        # Set weights for edges
        adjmatLogWEIGHT = adjmatLog.copy()
        np.fill_diagonal(adjmatLogWEIGHT.values, 0)
        adjmatLogWEIGHT = pd.DataFrame(index=adjmatLog.index.values, data=MinMaxScaler(feature_range=(0, 20)).fit_transform(adjmatLogWEIGHT), columns=adjmatLog.columns)

        # Set size for node
        IA, IB = ismember(adjmatLog.columns, self.results['counts'][:, 0])
        node_size = np.repeat(node_size_limits[0], len(adjmatLog.columns))
        node_size[IA] = hnstats._scale_weights(self.results['counts'][IB, 1], node_size_limits)

        # Make new graph (G) and add properties to nodes
        if not directed:
            adjmatLog = to_undirected(adjmatLog, method='logp')
            G = nx.Graph()
        else:
            G = nx.DiGraph(directed=True)

        # Color node using network-clustering
        # if node_color=='cluster':
            # _, labx = network.cluster(G.to_undirected())
        # else:
            # labx = self.results['labx']
        # Make sure the labx are correct if previously black/white list filtering was performed
        # IA,_ = ismember(self.results['simmatLogP'].columns, adjmatLog.columns)
        # labx = label_encoder.fit_transform(labx[IA])
        labx = label_encoder.fit_transform(labx)

        for i in range(0, adjmatLog.shape[0]):
            G.add_node(adjmatLog.index.values[i], node_size=node_size[i], node_label=labx[i])
        # Add properties to edges
        np.fill_diagonal(adjmatLog.values, 0)
        for i in range(0, adjmatLog.shape[0]):
            idx=np.where(adjmatLog.iloc[i, :]>0)[0]
            labels=adjmatLog.iloc[i, idx]
            labelsWEIGHT=adjmatLogWEIGHT.iloc[i, idx]

            for k in range(0, len(labels)):
                G.add_edge(labels.index[k], adjmatLog.index[i], weight=labels.values[k].astype(int), capacity=labelsWEIGHT.values[k])

        edges = G.edges()
        edge_weights = np.array([G[u][v]['weight'] for u, v in edges])
        edge_weights = MinMaxScaler(feature_range=(0.5, 8)).fit_transform(edge_weights.reshape(-1, 1)).flatten()

        # # Cluster
        if node_color=='cluster':
            _, labx = network.cluster(G.to_undirected())
        else:
            labx = label_encoder.fit_transform(labx)

        # G = nx.DiGraph() # Directed graph
        # Layout
        if isinstance(pos, type(None)):
            pos = nx.fruchterman_reingold_layout(G, weight='edge_weights', k=config['dist_between_nodes'], scale=config['scale'], iterations=config['iterations'])
        else:
            pos = network.graphlayout(G, pos=pos, scale=config['scale'], layout=config['layout'])

        # pos = nx.spring_layout(G, weight='edge_weights', k=config['dist_between_nodes'], scale=config['scale'], iterations=config['iterations'])
        # Boot-up figure
        if showfig or (not isinstance(config['savepath'], type(None))):
            [fig, ax] = plt.subplots(figsize=figsize)
            options = {
                # 'node_color': 'grey',
                'arrowsize': 12,
                'font_size': 18,
                'font_color': 'black'}
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
        Gout['G'] = G
        Gout['labx'] = labx
        Gout['pos'] = pos
        return(Gout)

    # Make plot of the association_learning
    def heatmap(self, summarize=False, cluster=False, figsize=[15, 15], savepath=None, threshold=None, white_list=None, black_list=None, min_edges=None, verbose=3):
        """Plot static heatmap.

        Description
        -----------
        A heatmap can be of use when the results becomes too large to plot in a network.

        Parameters
        ----------
        self : Object
            The output of .association_learning()
        summarize : bool, (default: False)
            Show the results based on categoric or label-specific associations.
            True: Summrize based on the categories
            False: All associations across labels
        cluster : Bool, optional
            Cluster before making heatmap. The default is False.
        figsize : typle, optional
            Figure size. The default is [15, 10].
        savepath : Bool, optional
            saveingpath. The default is None.
        threshold : int (default : None)
            Associations (edges) are filtered based on the -log10(P) > threshold. threshold should range between 0 and maximum value of -log10(P).
        black_list : List or None (default : None)
            If a list of edges is provided as black_list, they are excluded from the search and the resulting model will not contain any of those edges.
        white_list : List or None (default : None)
            If a list of edges is provided as white_list, the search is limited to those edges. The resulting model will then only contain edges that are in white_list.
        min_edges : int (default : None)
            Edges are only shown if a node has at least min_edges.
        verbose : int, optional
            Verbosity. The default is 3.

        Returns
        -------
        None.

        """
        if verbose>=3: print('[hnet] >Building static heatmap.')
        status = self._check_results()
        if not status: return None
        # Retrieve data
        _, simmatLogP, labx = self._get_adjmat(summarize)

        # adjmatLog = self.results['simmatLogP'].copy()
        # Filter adjacency matrix on blacklist/whitelist and/or threshold
        adjmatLog, labx = hnstats._filter_adjmat(simmatLogP, labx, threshold=threshold, min_edges=min_edges, white_list=white_list, black_list=black_list, verbose=verbose)
        if adjmatLog.values.flatten().sum()==0:
            if verbose>=3: print('[hnet] >Nothing to plot.')
            return None

        # Set savepath and filename
        savepath = hnstats._path_correct(savepath, filename='hnet_heatmap', ext='.png')

        try:
            savepath1=''
            if savepath is not None:
                [getdir, getname, getext]=hnstats._path_split(savepath)
                if getname=='': getname='heatmap'
                savepath1 = os.path.join(getdir, getname + '_logP' + getext)

            if cluster:
                fig1=imagesc.cluster(adjmatLog.fillna(value=0).values, row_labels=adjmatLog.index.values, col_labels=adjmatLog.columns.values, cmap='Reds', figsize=figsize)
            else:
                fig1=imagesc.plot(adjmatLog.fillna(value=0).values, row_labels=adjmatLog.index.values, col_labels=adjmatLog.columns.values, cmap='Reds', figsize=figsize)

            if savepath is not None:
                if verbose>=3: print('[hnet] >Saving heatmap..')
                _ = savefig(fig1, savepath1, transp=True)
        except:
            print('[hnet] >Error: Heatmap failed. Try cluster=False')

    # Extract combined rules from association_learning
    def combined_rules(self, simmatP=None, labx=None, verbose=3):
        """Association testing and combining Pvalues using fishers-method.

        Description
        -----------
        Multiple variables (antecedents) can be associated to a single variable (consequent).
        To test the significance of combined associations we used fishers-method. The strongest connection will be sorted on top.

        Parameters
        ----------
        simmatP : matrix
            simmilarity matrix
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
            Specific label names in the 'from' category.
        consequents
            Specific label names that are the result of the antecedents.
        Pfisher
            Combined P-value

        Examples
        --------
        >>> from hnet import hnet
        >>> hn = hnet()
        >>> df = hn.import_example('sprinkler')
        >>> hn.association_learning(df)
        >>> hn.combined_rules()
        >>> print(hn.rules)

        """
        if simmatP is None:
            if self.results.get('simmatP', None) is None: raise ValueError('[hnet] >Error: Input requires the result from the association_learning() function.')
            simmatP = self.results['simmatP']
        if labx is None:
            labx = self.results['labx']

        df_rules = pd.DataFrame(index=np.arange(0, simmatP.shape[0]), columns=['antecedents_labx', 'antecedents', 'consequents', 'Pfisher'])
        df_rules['consequents'] = simmatP.index.values

        for i in tqdm(range(0, simmatP.shape[0]), disable=(True if verbose==0 else False)):
            idx = np.where(simmatP.iloc[i, :]<1)[0]
            # Remove self
            idx = np.setdiff1d(idx, i)
            # Store rules
            df_rules['antecedents'].iloc[i] = list(simmatP.iloc[i, idx].index)
            df_rules['antecedents_labx'].iloc[i] = labx[idx]
            # Combine pvalues
            df_rules['Pfisher'].iloc[i] = combine_pvalues(simmatP.iloc[i, idx].values, method='fisher')[1]

        # Keep only lines with pvalues
        df_rules.dropna(how='any', subset=['Pfisher'], inplace=True)
        # Sort
        df_rules.sort_values(by=['Pfisher'], ascending=True, inplace=True)
        df_rules.reset_index(inplace=True, drop=True)
        # Return
        return(df_rules)

    def import_example(self, data='titanic', url=None, sep=',', verbose=3):
        """Import example dataset from github source.

        Description
        -----------
        Import one of the few datasets from github source or specify your own download url link.

        Parameters
        ----------
        data : str
            Name of datasets: 'sprinkler', 'titanic', 'student', 'fifa', 'cancer', 'waterpump', 'retail'
        url : str
            url link to to dataset.
        verbose : int, (default: 3)
            Print message to screen.

        Returns
        -------
        pd.DataFrame()
            Dataset containing mixed features.

        """
        return import_example(data=data, url=url, sep=sep, verbose=verbose)

    # Save model
    def save(self, filepath='hnet_model.pkl', overwrite=False, verbose=3):
        """Save learned model in pickle file.

        Parameters
        ----------
        filepath : str, (default: 'hnet_model.pkl')
            Pathname to store pickle files.
        overwrite : bool, (default=False)
            Overwite file if exists.
        verbose : int, optional
            Show message. A higher number gives more informatie. The default is 3.

        Returns
        -------
        bool : [True, False]
            Status whether the file is saved.

        """
        if (filepath is None) or (filepath==''):
            filepath = 'hnet_model.pkl'
        if filepath[-4:] != '.pkl':
            filepath = filepath + '.pkl'
        # Store data
        storedata = {}
        storedata['results'] = self.results
        storedata['alpha'] = self.alpha
        storedata['k'] = self.k
        storedata['y_min'] = self.y_min
        storedata['multtest'] = self.multtest
        storedata['dtypes'] = self.dtypes
        storedata['specificity'] = self.specificity
        storedata['perc_min_num'] = self.perc_min_num
        storedata['dropna'] = self.dropna
        storedata['fillna'] = self.fillna
        storedata['excl_background'] = self.excl_background
        # Save
        status = pypickle.save(filepath, storedata, overwrite=overwrite, verbose=verbose)
        if verbose>=3: print('[hnet] >Saving.. %s' %(status))
        # return
        return status

    # Load model.
    def load(self, filepath='hnet_model.pkl', verbose=3):
        """Load learned model.

        Parameters
        ----------
        filepath : str
            Pathname to stored pickle files.
        verbose : int, optional
            Show message. A higher number gives more information. The default is 3.

        Returns
        -------
        Object.

        """
        if (filepath is None) or (filepath==''):
            filepath = 'hnet_model.pkl'
        if filepath[-4:]!='.pkl':
            filepath = filepath + '.pkl'
        # Load
        storedata = pypickle.load(filepath, verbose=verbose)
        # Store in self.
        if storedata is not None:
            self.results = storedata['results']
            self.alpha = storedata['alpha']
            self.k = storedata['k']
            self.y_min = storedata['y_min']
            self.multtest = storedata['multtest']
            self.dtypes = storedata['dtypes']
            self.specificity = storedata['specificity']
            self.perc_min_num = storedata['perc_min_num']
            self.dropna = storedata['dropna']
            self.fillna = storedata['fillna']
            self.excl_background = storedata['excl_background']
            if verbose>=3: print('[hnet] >Loading succesfull!')
            # Return results
            return self.results
        else:
            if verbose>=2: print('[hnet] >WARNING: Could not load data.')

    # Check results
    def _check_results(self, verbose=3):
        status = True
        if not hasattr(self, 'results'):
            if verbose>=3: print('[hnet] >Nothing to plot. Try to run hnet with first with: hn.association_learning()')
            status = False
        elif self.results['simmatLogP'].empty:
            if verbose>=3: print('[hnet] >Nothing to plot. No associations were detected.')
            status = False

        return status

    def _clean(self, verbose=3):
        # Clean readily fitted models to ensure correct results.
        if hasattr(self, 'results'):
            if verbose>=3: print('[hnet] >Cleaning previous fitted model results..')
            if hasattr(self, 'results'): del self.results

# %% Store results


def _store(simmatP, adjmatLog, GsimmatP, GsimmatLogP, labx, df, nr_succes_pop_n, dtypes, rules, feat_importance):
    out = {}
    out['simmatP'] = simmatP
    out['simmatLogP'] = adjmatLog
    out['simmatP_cat'] = GsimmatP
    out['simmatLogP_cat'] = GsimmatLogP
    out['labx'] = labx.astype(str)
    out['dtypes'] = np.array(list(zip(df.columns.values.astype(str), dtypes)))
    out['counts'] = nr_succes_pop_n
    out['rules'] = rules
    out['feat_importance'] = feat_importance
    return out


# %% Import example dataset from github.
def import_example(data='titanic', url=None, sep=',', verbose=3):
    """Import example dataset from github source.

    Description
    -----------
    Import one of the few datasets from github source or specify your own download url link.

    Parameters
    ----------
    data : str
        Name of datasets: 'sprinkler', 'titanic', 'student', 'fifa', 'cancer', 'waterpump', 'retail'
    url : str
        url link to to dataset.
    verbose : int, (default: 3)
        Print message to screen.

    Returns
    -------
    pd.DataFrame()
        Dataset containing mixed features.

    """
    if url is None:
        if data=='sprinkler':
            url='https://erdogant.github.io/datasets/sprinkler.zip'
        elif data=='titanic':
            url='https://erdogant.github.io/datasets/titanic_train.zip'
        elif data=='student':
            url='https://erdogant.github.io/datasets/student_train.zip'
        elif data=='cancer':
            url='https://erdogant.github.io/datasets/cancer_dataset.zip'
        elif data=='fifa':
            url='https://erdogant.github.io/datasets/FIFA_2018.zip'
        elif data=='waterpump':
            url='https://erdogant.github.io/datasets/waterpump/waterpump_test.zip'
        elif data=='retail':
            url='https://erdogant.github.io/datasets/marketing_data_online_retail_small.zip'
    else:
        data = wget.filename_from_url(url)

    if url is None:
        if verbose>=3: print('[hnet] >Nothing to download.')
        return None

    curpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    PATH_TO_DATA = os.path.join(curpath, wget.filename_from_url(url))
    if not os.path.isdir(curpath):
        os.makedirs(curpath, exist_ok=True)

    # Check file exists.
    if not os.path.isfile(PATH_TO_DATA):
        if verbose>=3: print('[hnet] >Downloading [%s] dataset from github source..' %(data))
        wget.download(url, curpath)

    # Import local dataset
    if verbose>=3: print('[hnet] >Import dataset [%s]' %(data))
    df = pd.read_csv(PATH_TO_DATA, sep=sep)
    # Return
    return df


# %% Compute fit
def enrichment(df, y, y_min=None, alpha=0.05, multtest='holm', dtypes='pandas', specificity='medium', excl_background=None, verbose=3):
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
    excl_background : String (default : None)
        Name to exclude from the background.
        Example: ['0.0']: To remove categorical values with label 0
    verbose : int, optional
        Print message to screen. The higher the number, the more details. The default is 3.

    Returns
    -------
    pd.DataFrame() with the following columns:
    category_label : str
        Label of the category.
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
    >>> import hnet as hn
    >>> df = hn.import_example('titanic')
    >>> y = df['Survived'].values
    >>> out = hn.enrichment(df, y)

    """
    assert isinstance(df, pd.DataFrame), 'Data must be of type pd.DataFrame()'
    assert len(y)==df.shape[0], 'Length of [df] and [y] must be equal'
    assert 'numpy' in str(type(y)), 'y must be of type numpy array'

    # DECLARATIONS
    config = {}
    config['verbose'] = verbose
    config['alpha'] = alpha
    config['multtest'] = multtest
    config['specificity'] = specificity

    if config['verbose']>=3: print('[hnet] >Start making fit..')
    df.columns = df.columns.astype(str)

    # [df, df_onehot, dtypes] = hnstats._preprocessing(df, dtypes=dtypes, y_min=y_min, perc_min_num=perc_min_num, excl_background=excl_background, verbose=verbose)
    df, dtypes, excl_background = hnstats._bool_processesing(df, dtypes, excl_background=excl_background, verbose=verbose)

    # Set y as string
    y = df2onehot.set_y(y, y_min=y_min, verbose=config['verbose'])
    # Determine dtypes for columns
    df, dtypes = df2onehot.set_dtypes(df, dtypes, verbose=config['verbose'])
    # Compute fit
    out = hnstats._compute_significance(df, y, dtypes, specificity=config['specificity'], verbose=config['verbose'])
    # Multiple test correction
    out = hnstats._multipletestcorrection(out, config['multtest'], verbose=config['verbose'])
    # Keep only significant ones
    out = hnstats._filter_significance(out, config['alpha'], multtest)
    # Make dataframe
    out = pd.DataFrame(out)
    # Return
    if config['verbose']>=3: print('[hnet] >Fin')
    return(out)


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
        raise ValueError('Adjacency matrix must have similar number of rows and columns! Re-run HNet with dropna=False!')

    if verbose>=3: print('[hnet] >Make adjacency matrix undirected..')
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
        Difference of the input network compared to the second.
        0 = No edge
        1 = No difference between networks
        2 = Addition of edge in the first input network compared to the second
       -1 = Depliction of edge in the first netwwork compared to the second

    """
    scores, adjmat_diff = network.compare_networks(adjmat_true,
                                                   adjmat_pred,
                                                   pos=pos,
                                                   showfig=showfig,
                                                   width=width,
                                                   height=height,
                                                   verbose=verbose)
    return(scores, adjmat_diff)


# %% Do the math
def _do_the_math(df, X_comb, dtypes, X_labx, simmatP, simmat_labx, i, specificity, y_min, verbose=3):
    count = 0
    # Get response variable to test association
    y = X_comb.iloc[:, i].values.astype(str)
    # Get column name
    colname = X_comb.columns[i]
    # Default output is nan
    out = [colname, np.nan]
    # Set max-string length for verbosity message
    maxstr = np.minimum(X_comb.columns.str.len().max(), 40)
    # Do math if response variable has more then 1 option
    if len(np.unique(y))>1:
        if verbose>=4: print('[hnet] >Working on [%s]%s ' %(X_comb.columns[i], '.' * (maxstr - len(X_comb.columns[i]))), end='')
        # Remove columns if it belongs to the same categorical subgroup; these can never overlap!
        Iloc = ~np.isin(df.columns, X_labx[i])
        # Compute fit
        dfout = enrichment(df.loc[:, Iloc], y, y_min=y_min, alpha=1, multtest=None, dtypes=dtypes[Iloc], specificity=specificity, verbose=0)
        # Count
        count=count + dfout.shape[0]
        # Match with dataframe and store
        if not dfout.empty:
            # Column names
            idx = np.where(dfout['category_label'].isna())[0]
            catnames = dfout['category_name']
            colnames = catnames + '_' + dfout['category_label'].astype(str)
            colnames[idx] = catnames[idx].values
            # Add new column and index
            simmatP, simmat_labx = hnstats._addcolumns(simmatP, colnames, simmat_labx, catnames)
            # Store values
            IA, IB = ismember(simmatP.index.values.astype(str), colnames.values.astype(str))
            simmatP.loc[colname, IA] = dfout['Padj'].iloc[IB].values
            # Count nr. successes
            out = [colname, X_comb.iloc[:, i].sum() / X_comb.shape[0]]
            # showprogress
            if verbose>=4: print('[%g]' %(len(IB)), end='')
    else:
        if verbose>=4: print('[hnet] >Skipping [%s] because length of unique values=1' %(X_comb.columns[i]), end='')

    if verbose>=4: print('')
    # Return
    return(out, simmatP, simmat_labx)
