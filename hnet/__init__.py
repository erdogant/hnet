from hnet.hnet import hnet

from hnet.hnet import (
    enrichment,
    compare_networks,
    to_undirected,
    import_example,
    )

from hnet.adjmat_vec import (
    vec2adjmat,
    adjmat2vec,
    )

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '1.1.12'

# module level doc-string
__doc__ = """
HNET - Graphical Hypergeometric networks.
=====================================================================

Description
-----------
Creation of networks from datasets with mixed datatypes and with unknown function.
Input datasets can range from generic dataframes to nested data structures with lists, missing values and enumerations.
HNet (graphical Hypergeometric Networks) is a method where associations across variables are tested for significance by statistical inference.
The aim is to determine a network with significant associations that can shed light on the complex relationships across variables.

Examples
--------
>>> # Initialize hnet with default settings
>>> from hnet import hnet
>>> hn = hnet()
>>> # Load example dataset
>>> df = hn.import_example('titanic')

>>> # Structure learning
>>> out = hn.association_learning(df)
>>>
>>> # Plot dynamic graphs
>>> G = hn.d3graph()
>>> G = hn.d3heatmap()
>>>
>>> # Plot static graph
>>> G = hn.plot()
>>>
>>> # Plot heatmap
>>> hn.heatmap(cluster=True)
>>>
>>> # Plot feature importance
>>> hn.plot_feat_importance()
>>>
>>> # Examine differences between models
>>> import hnet
>>> scores, adjmat = hnet.compare_networks(out['simmatP'], out['simmatP'], showfig=True)
>>> adjmat_undirected = hnet.to_undirected(out['simmatLogP'])
>>>

References
----------
* Blog: https://towardsdatascience.com/explore-and-understand-your-data-with-a-network-of-significant-associations-9a03cf79d254
* Github: https://github.com/erdogant/hnet
* Documentation: https://erdogant.github.io/hnet/
* Article: https://arxiv.org/abs/2005.04679

"""
