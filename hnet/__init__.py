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

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '0.1.7'

# module level doc-string
__doc__ = """
HNET - Graphical Hypergeometric networks.
=====================================================================

Description
-----------
Creating networks from datasets with mixed datatypes return
by an unknown function. These datasets can range from generic dataframes to
nested data structures with lists, missing values and enumerations.
I solved this problem to minimize the amount of configurations required
while still gaining many benefits of having schemas available.

The response variable (y) should be a vector with the same number of samples
as for the input data. For each column in the dataframe significance is
assessed for the labels in a two-class approach (y=1 vs y!=1).
Significane is assessed one tailed; only the fit for y=1 with an
overrepresentation. Hypergeometric test is used for catagorical values
Wilcoxen rank-sum test for numerical values.


Example
-------

>>> import hnet
>>> model = hnet.fit(df)
>>> G = hnet.plot(model)
>>> G = hnet.heatmap(model)
>>> G = hnet.d3graph(model)
>>> [scores, adjmat] = hnet.compare_networks(model['adjmat'], model['adjmat'])
>>> rules = hnet.combined_rules(model)
>>> adjmatSymmetric = hnet.to_symmetric(model)
"""
