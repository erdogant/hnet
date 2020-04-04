from hnet.hnet import (
    fit,
    enrichment,
	plot,
	heatmap,
	d3graph,
	combined_rules,
	compare_networks,
	to_symmetric,
	import_example,
)

from hnet.utils.adjmat_vec import (
    vec2adjmat,
    adjmat2vec,
    )

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '0.1.8'

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


Example
-------

>>> import hnet
>>> df = hnet.import_example('student')
>>> model = hnet.fit(df)
>>> G = hnet.plot(model)
>>> G = hnet.heatmap(model)
>>> G = hnet.d3graph(model)
>>> [scores, adjmat] = hnet.compare_networks(model['adjmat'], model['adjmat'])
>>> rules = hnet.combined_rules(model)
>>> adjmatSymmetric = hnet.to_symmetric(model)
"""
