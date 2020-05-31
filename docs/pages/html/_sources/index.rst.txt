HNet documentation!
========================

*-HNet Graphical Hypergeometric Networks-*

Real-world data sets often contain measurements with both continues and categorical values for the same sample. Despite the availability of many libraries, data sets with mixed data types require intensive pre-processing steps, and it remains a challenge to describe the relationships of one variable on another. The data understanding part is crucial but without making any assumptions on the model form, the search space is super-exponential in the number of variables and therefore not a common practice.

We propose graphical hypergeometric networks (HNet), a method where associations across variables are tested for significance by statistical inference. 

HNet learns the Association from datasets with mixed datatypes and with unknown function. Input datasets can range from generic dataframes to nested data structures with lists, missing values and enumerations. The aim is to determine a network with significant associations that can shed light on the complex relationships across variables.


Content
=======
.. toctree::
   :maxdepth: 1
   :caption: Background

   Abstract


.. toctree::
  :maxdepth: 1
  :caption: Installation

  Installation


.. toctree::
  :maxdepth: 1
  :caption: Input/output

  input_output


.. toctree::
  :maxdepth: 3
  :caption: Algorithm

  Methods
  

.. toctree::
  :maxdepth: 1
  :caption: Plots

  Plots
  

.. toctree::
  :maxdepth: 1
  :caption: Examples

  Examples
  Use Cases


.. toctree::
  :maxdepth: 1
  :caption: Code Documentation

  Coding quality
  hnet.hnet
  arXiv



Code and paper
------------------------------

Please cite ``hnet`` in your publications if this is useful for your research.

* arXiv: `arXiv <https://arxiv.org/abs/2005.04679>`_
* Paper: `Paper <https://arxiv.org/pdf/2005.04679>`_
* Github: `erdogant/hnet <https://github.com/erdogant/hnet/>`_
* Please report bugs, issues and feature extensions on the github page.

Here is the BibTeX entry:

.. code-block:: python

	@misc{taskesen2020hnet,
	    title={HNet: Graphical Hypergeometric Networks},
	    author={Erdogan Taskesen},
	    year={2020},
	    eprint={2005.04679},
	    archivePrefix={arXiv},
	    primaryClass={cs.LG}
	}


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
