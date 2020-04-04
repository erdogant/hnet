.. _code_directive:

-------------------------------------

Quickstart
''''''''''

Installing ``hnet``

.. code:: bash

    pip install hnet


A quick example how to learn the structure on a given dataset.

.. code:: python

    # Import library
    import hnet

    # Import data:
    df = hnet.import_example()

    # Learn structure on the data
    model = hnet.fit(df)

    # Plot results
    G = hnet.plot(model)

    # Plot heatmap
    G = hnet.heatmap(model)

    # Plot interactive plot
    G = hnet.d3graph(model)


Installation
''''''''''''

Create environment
------------------

Example how to install ``hnet`` via ``pip`` in an isolated Python environment:

.. code-block:: python

    conda create -n env_hnet python=3.6
    conda activate env_hnet


The installation of ``hnet`` from pypi:

.. code-block:: console

    pip install hnet


Install latest beta version from github source:

.. code-block:: console

    pip install git+https://github.com/erdogant/hnet

