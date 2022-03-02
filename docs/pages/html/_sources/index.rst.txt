HNet documentation!
========================

*-HNet Graphical Hypergeometric Networks-*

.. tip::
	`Medium blog: HNet <https://towardsdatascience.com/explore-and-understand-your-data-with-a-network-of-significant-associations-9a03cf79d254>`_


Real-world data sets often contain measurements with both continues and categorical values for the same sample. Despite the availability of many libraries, data sets with mixed data types require intensive pre-processing steps, and it remains a challenge to describe the relationships of one variable on another. The data understanding part is crucial but without making any assumptions on the model form, the search space is super-exponential in the number of variables and therefore not a common practice.

We propose graphical hypergeometric networks (HNet), a method where associations across variables are tested for significance by statistical inference. 

HNet learns the Association from datasets with mixed datatypes and with unknown function. Input datasets can range from generic dataframes to nested data structures with lists, missing values and enumerations. The aim is to determine a network with significant associations that can shed light on the complex relationships across variables.

HNet can be used for all kind of datasets that contain features such as categorical, boolean, and/or continuous values.

    Your goal can be for example:
        1. Explore the complex associations between your variables.
        2. Explain your clusters by enrichment of the meta-data.
        3. Transform your feature space into network graph and/or dissimilarity matrix that can be used for further analysis.



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
  :caption: Documentation

  Blog
  Coding quality
  hnet.hnet

* :ref:`genindex`



Source code and issue tracker
------------------------------

`Github hnet <https://github.com/erdogant/hnet/>`_.
Please report bugs, issues and feature extensions there.


Citing *hnet*
----------------

The bibtex can be found in the right side menu at the `github page <https://github.com/erdogant/hnet/>`_.


Sponsor this project
------------------------------

If you like this project, **star** this repo and become a **sponsor**!
Read more why this is important on my sponsor page!

.. raw:: html

	<iframe src="https://github.com/sponsors/erdogant/button" title="Sponsor erdogant" height="35" width="116" style="border: 0;"></iframe>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. raw:: html

	<hr>
	<center>
		<script async type="text/javascript" src="//cdn.carbonads.com/carbon.js?serve=CEADP27U&placement=erdogantgithubio" id="_carbonads_js"></script>
	</center>
	<hr>

