# --------------------------------------------------------------------------
# hnet initialization
# --------------------------------------------------------------------------
import logging

# Optional TRACE level (also used by set_logger)
if not hasattr(logging, 'TRACE'):
    logging.TRACE = 5
    logging.addLevelName(logging.TRACE, 'TRACE')

# Package logger. Default = INFO. set_logger / get_logger live in hnet.utils
# and are re-exported below so users can do: import hnet as hn; hn.set_logger('info')
logger = logging.getLogger('hnet')
_log_handler = logging.StreamHandler()
_fmt = '[{asctime}] [{name}] [{levelname}] {message}'
_formatter = logging.Formatter(fmt=_fmt, style='{', datefmt='%d-%m-%Y %H:%M:%S')
_log_handler.setFormatter(_formatter)
_log_handler.setLevel(logging.DEBUG)
if not logger.handlers:  # avoid duplicate handlers on re-import
    logger.addHandler(_log_handler)
logger.setLevel(logging.INFO)
logger.propagate = False

# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
from hnet.hnet import hnet

from hnet.hnet import (
    enrichment,
    compare_networks,
    to_undirected,
)

from hnet.adjmat_vec import (
    vec2adjmat,
    adjmat2vec,
)

from hnet.utils import set_logger, get_logger, log_prefix

from datazets import get as import_example

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '1.3.4'

# module level doc-string
__doc__ = """
HNET - Graphical Hypergeometric networks.
=====================================================================

Creation of networks from datasets with mixed datatypes and with unknown function.
Input datasets can range from generic dataframes to nested data structures with lists, missing values and enumerations.
HNet (graphical Hypergeometric Networks) is a method where associations across variables are tested for significance by statistical inference.
The aim is to determine a network with significant associations that can shed light on the complex relationships across variables.

Examples
--------
>>> # Import library and set verbosity
>>> import hnet as hn
>>> hn.set_logger('info')   # or 'warning', 'error', 'debug', 'trace', None
>>>
>>> # Initialize hnet with default settings
>>> model = hn.hnet()
>>> # Load example dataset
>>> df = model.import_example('titanic')
>>>
>>> # Structure learning
>>> out = model.association_learning(df)
>>>
>>> # Plot dynamic graphs
>>> G = model.d3graph()
>>> G = model.d3heatmap()
>>>
>>> # Plot static graph
>>> G = model.plot()
>>>
>>> # Plot heatmap
>>> model.heatmap(cluster=True)
>>>
>>> # Plot feature importance
>>> model.plot_feat_importance()
>>>
>>> # Examine differences between models
>>> scores, adjmat = hn.compare_networks(out['simmatP'], out['simmatP'], showfig=True)
>>> adjmat_undirected = hn.to_undirected(out['simmatLogP'])
>>>

References
----------
* Blog: https://erdogant.github.io/hnet/pages/html/Documentation.html#blog
* Github: https://github.com/erdogant/hnet
* Documentation: https://erdogant.github.io/hnet/
* Article: https://arxiv.org/abs/2005.04679

"""
