.. include:: add_top.add


Quickstart
''''''''''

Installing ``hnet``

.. code:: bash

    pip install hnet


A quick example how to learn the association on a given dataset.

.. code:: python

    # Load library
    from hnet import hnet
    
    # Import library with default settings
    hn = hnet()

    # Import data:
    df = hn.import_example('titanic')

    # Learn association on the data
    results = hn.association_learning(df)

    # Plot dynamic graph
    G_dynamic = hn.d3graph()

    # Plot static graph
    G_static = hn.plot()
    
    # Plot heatmap
    P_heatmap = hn.heatmap(cluster=True)


Installation
''''''''''''

**Create environment**


Example how to install ``hnet`` via ``pip`` in an isolated Python environment:

.. code-block:: python

    conda create -n env_hnet python=3.7
    conda activate env_hnet


The installation of ``hnet`` from pypi:

.. code-block:: console

    pip install hnet




.. include:: add_bottom.add