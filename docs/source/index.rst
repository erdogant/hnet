HNet documentation!
========================

|python| |pypi| |docs| |stars| |LOC| |downloads_month| |downloads_total| |license| |forks| |open issues| |project status| |medium| |colab| |DOI| |repo-size| |donate|

.. include:: add_top.add


.. tip::
	`Read the details and usage of HNet in the Medium blog! <https://towardsdatascience.com/explore-and-understand-your-data-with-a-network-of-significant-associations-9a03cf79d254>`_


Real-world data sets often contain measurements with both continues and categorical values for the same sample. Despite the availability of many libraries, data sets with mixed data types require intensive pre-processing steps, and it remains a challenge to describe the relationships of one variable on another. The data understanding part is crucial but without making any assumptions on the model form, the search space is super-exponential in the number of variables and therefore not a common practice.
We propose graphical hypergeometric networks (HNet), a method where associations across variables are tested for significance by statistical inference. 
HNet learns the Association from datasets with mixed datatypes and with unknown function. Input datasets can range from generic dataframes to nested data structures with lists, missing values and enumerations. The aim is to determine a network with significant associations that can shed light on the complex relationships across variables.
HNet can be used for all kind of datasets that contain features such as categorical, boolean, and/or continuous values.

Use HNET if your goal is:

* 1. Explore the complex associations between your variables.
* 2. Explain your clusters by enrichment of the meta-data.
* 3. Transform your feature space into network graph and/or dissimilarity matrix that can be used for further analysis.


You contribution is important
==============================
If you ❤️ this project, **star** this repo at the `github page <https://github.com/erdogant/hnet/>`_ and have a look at the `sponser page <https://erdogant.github.io/hnet/pages/html/Documentation.html>`_!


Github
======
Please report bugs, issues and feature extensions at `github <https://github.com/erdogant/hnet/>`_.



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

  Documentation
  Coding quality
  hnet.hnet

* :ref:`genindex`



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. |python| image:: https://img.shields.io/pypi/pyversions/hnet.svg
    :alt: |Python
    :target: https://erdogant.github.io/hnet/

.. |pypi| image:: https://img.shields.io/pypi/v/hnet.svg
    :alt: |Python Version
    :target: https://pypi.org/project/hnet/

.. |docs| image:: https://img.shields.io/badge/Sphinx-Docs-blue.svg
    :alt: Sphinx documentation
    :target: https://erdogant.github.io/hnet/

.. |stars| image:: https://img.shields.io/github/stars/erdogant/hnet
    :alt: Stars
    :target: https://img.shields.io/github/stars/erdogant/hnet

.. |LOC| image:: https://sloc.xyz/github/erdogant/hnet/?category=code
    :alt: lines of code
    :target: https://github.com/erdogant/hnet

.. |downloads_month| image:: https://static.pepy.tech/personalized-badge/hnet?period=month&units=international_system&left_color=grey&right_color=brightgreen&left_text=PyPI%20downloads/month
    :alt: Downloads per month
    :target: https://pepy.tech/project/hnet

.. |downloads_total| image:: https://static.pepy.tech/personalized-badge/hnet?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads
    :alt: Downloads in total
    :target: https://pepy.tech/project/hnet

.. |license| image:: https://img.shields.io/badge/license-MIT-green.svg
    :alt: License
    :target: https://github.com/erdogant/hnet/blob/master/LICENSE

.. |forks| image:: https://img.shields.io/github/forks/erdogant/hnet.svg
    :alt: Github Forks
    :target: https://github.com/erdogant/hnet/network

.. |open issues| image:: https://img.shields.io/github/issues/erdogant/hnet.svg
    :alt: Open Issues
    :target: https://github.com/erdogant/hnet/issues

.. |project status| image:: http://www.repostatus.org/badges/latest/active.svg
    :alt: Project Status
    :target: http://www.repostatus.org/#active

.. |medium| image:: https://img.shields.io/badge/Medium-Blog-green.svg
    :alt: Medium Blog
    :target: https://erdogant.github.io/hnet/pages/html/Documentation.html#medium-blog

.. |donate| image:: https://img.shields.io/badge/Support%20this%20project-grey.svg?logo=github%20sponsors
    :alt: donate
    :target: https://erdogant.github.io/hnet/pages/html/Documentation.html#

.. |colab| image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Colab example
    :target: https://erdogant.github.io/hnet/pages/html/Documentation.html#colab-notebook

.. |DOI| image:: https://zenodo.org/badge/226647104.svg
    :alt: Cite
    :target: https://zenodo.org/badge/latestdoi/226647104

.. |repo-size| image:: https://img.shields.io/github/repo-size/erdogant/hnet
    :alt: repo-size
    :target: https://img.shields.io/github/repo-size/erdogant/hnet


.. include:: add_bottom.add