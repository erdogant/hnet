from hnet.hnet import (
    fit,
    enrichment,
	plot_heatmap,
	plot_network,
	plot_d3graph,
	combined_rules,
	compare_networks,
	to_symmetric,
)

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
#__version__ = '0.1.0'

# Automatic version control
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


# module level doc-string
__doc__ = """
HNET - Hypergeometric networks. Detection of associations for datasets with mixed datatypes with unknown function.
=====================================================================

**hnet** 
See README.md file for more information.

"""
