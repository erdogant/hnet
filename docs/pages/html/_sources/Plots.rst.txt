.. _code_directive:

-------------------------------------

Network Graphs
'''''''''''''''

Dynamic graph representations are created using ``d3graph`` that allow to deeper examine the detected associations. Just like static graphs, the dynamic graph consists out of nodes and edges for which sizes and colours are adjusted accordingly. 
The advantage is that d3graph is an interactive and stand-alone network. The network is created with collision and charge parameters to ensure that nodes do not overlap. 

d3graph is developed as a stand-alone python library (https://github.com/erdogant/d3graph) which generates java script based on a set of user-defined or ``hnet`` parameters. The java script file is built on functionalities from the d3 javascript library (version 3). 


.. code-block:: bash

  pip install d3graph


In its simplest form, the input for d3graph is an adjacency matrix for which the elements indicate pairs of vertices are adjacent or not in the graph.


.. table::
  
  +-----------+--------+-----------+--------+-----------+
  |           | Node 1 | Node 2    | Node 3 | Node 4    |
  +===========+========+===========+========+===========+
  | Node 1    | False  | True      | True   | False     |
  +-----------+--------+-----------+--------+-----------+
  | Node 2    | False  | False     | False  | True      |
  +-----------+--------+-----------+--------+-----------+
  | Node 3    | False  | False     | False  | True      |
  +-----------+--------+-----------+--------+-----------+
  | Node 4    | False  | False     | False  | False     |
  +-----------+--------+-----------+--------+-----------+



d3graph - Dynamic network
''''''''''''''''''''''''''''''

.. code-block:: python

  from hnet import hnet
  # Load example
  df = hnet.import_example('sprinkler')

  # Association learning
  hn.association_learning(df)
  
  # Plot dynamic graph
  G_dynamic = hn.d3graph()


Each node contains a text-label, whereas the links of associated nodes can be highlighted when double clicked on the node of interest. Furthermore, each node involves a tooltip that can easily be adapted to display any of the underlying data. For deeper examination of the network, edges can be gradually broken on its weight using a slider. 

.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/sprinkler_example/index.html" height="500px" width="1000px", frameBorder="0"></iframe>



Heatmap
''''''''''

A heatmap can be of use when the results becomes too large to plot in a network.
Below is depicted a demonstration of plotting the results of ``hnet`` using a heatmap:

.. code-block:: python

  from hnet import hnet

  # Load example
  df = hnet.import_example('sprinkler')
  
  # Association learning
  hn.association_learning(df)

  # Plot heatmap
  hn.heatmap(cluster=True)


.. _schematic_overview:

.. figure:: ../figs/other/sprinkler_heatmap_clustered.png


d3eatmap - Dynamic heatmap
''''''''''''''''''''''''''''''

A heatmap can also be created using d3-javascript, where each cell ij represents an edge from vertex i to vertex j.
Given this two-dimensional representation of a graph, a natural visualization is to show the matrix!
However, the effectiveness of a matrix diagram is heavily dependent on the order of rows and columns: if related nodes are placed closed to each other, it is easier to identify clusters and bridges.
While path-following is harder in a matrix view than in a node-link diagram, matrices have other advantages.
As networks get large and highly connected, node-link diagrams often devolve into giant hairballs of line crossings.
Line crossings are impossible with matrix views. Matrix cells can also be encoded to show additional data; here color depicts clusters computed by a community-detection algorithm.
 
Below is depicted a demonstration of plotting the results of ``hnet`` using a d3heatmap:

.. code-block:: python
	
	# Generate the interactive heatmap
	G = hn.d3heatmap()


.. raw:: html

   <iframe src="https://erdogant.github.io/docs/titanic/d3heatmap/titanic_heatmap.html" height="1000px" width="100%", frameBorder="0"></iframe>


Feature importance
'''''''''''''''''''''

Feature importance can be plotted to analyze the involvement of the feature in the network.

.. code-block:: python

  from hnet import hnet

  # Load example
  df = hnet.import_example('titanic')
  
  # Association learning
  hn.association_learning(df)

  # Plot heatmap
  hn.plot_feat_importance()

The count of number of significance edges per node. We clearly see that **Parch_2** has the most significanly connected edges.
The coloring of the nodes is based on the catagory label. As an example, all **Parch** classes are labeled orange. Note that quantitative nodes
will have low/no significant edges.

.. _feat_importance_labels:

.. figure:: ../figs/other/feat_imp_1.png

Instead of counting individual node labels (as depicted previously), we can also count the total number of edges for a specific catagory.
Here we can clearly see that **SibSp** contains, in total, the most significant edges.

.. _feat_importance_cat:

.. figure:: ../figs/other/feat_imp_2.png

In the next figure we demonstrate the feature importance by a normalized significance.
The number of significant edges in the catagory labels is heavily influenced by the available labels.
As an example **SibSp** contains, in total, the most significant edges but this may be because it also contains the most labels.
In the following figure we correct for the number of labels per catagory.

.. _feat_importance_norm:

.. figure:: ../figs/other/feat_imp_3.png


Summarize to categories
'''''''''''''''''''''''''

If many associations are detected, a network plot can lead to a giant hairball, and a heatmap can become unreadable.
A function in ``hnet`` is implemented to summarize the associations towards categories. This would lead to generic insights.
As an example, in case of the **titanic** use-case, it will not describe wether **Parch 2** was associated with 4 **SibSp**
but instead it will describe wether **Parch** was significantly associated with **SibSp**. This is computed by using *Fishers* method.

.. code-block:: python

  from hnet import hnet

  # Load example
  df = hnet.import_example('titanic')
  
  # Association learning
  hn.association_learning(df)

  # Plot Network
  hn.d3graph(summarize=True)
  hn.plot(summarize=True)

  # Plot heatmap
  hn.d3heatmap(summarize=True)
  hn.heatmap(summarize=True)


.. _static_heatmap_summarize:

.. figure:: ../figs/other/titanic_summarize_static_heatmap.png


raw:: html

   <iframe src="https://erdogant.github.io/docs/titanic/d3heatmap/titanic_heatmap_summarize.html" height="1000px" width="100%", frameBorder="0"></iframe>

.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/titanic_example/titanic_summarize.html" height="1000px" width="100%", frameBorder="0"></iframe>


Comparing networks
''''''''''''''''''

Comparison of two networks based on two adjacency matrices. Both matrices should be of equal size and of type pandas DataFrame. The columns and rows between both matrices are matched if not ordered similarly.

Below is depicted a demonstration of comparing two networks that may have been the result of ``hnet`` using different parameter settings:


.. code-block:: python
  
  import hnet

  # Examine differences between models
  [scores, adjmat] = hnet.compare_networks(adjmat1, adjmat2)



Black/White listing in plots
'''''''''''''''''''''''''''''''''''

It is sometimes desired to remove variables from the plot to reduce complexity or to focus on specific associations.

Four methods of filtering are possible in ``hnet``

    * black_list : Excluded nodes form the plot. The resulting plot will not contain this node(s).
    * white_list : Only included the listed nodes in the plot. The resulting plot will be limited to the specified node(s).
    * threshold : Associations (edges) are filtered based on the -log10(P) > threshold. threshold should range between 0 and maximum value of -log10(P).
    * min_edges : Nodes are only shown if it contains at least a minimum number of edges.


Black listing in plot
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

  # In this example we will remove the node Age and SibSp.
  
  # d3graph
  hn = hn.d3graph(black_list=['Age', 'SibSp'])
  # Plot
  hn = hn.plot(black_list=['Age', 'SibSp'])
  # Heatmap
  hn = hn.heatmap(black_list=['Age', 'SibSp'])


White listing in plot
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

  # In this example we will keep only the node Survived and SibSp
  
  # d3graph
  hn = hn.d3graph(white_list=['Survived', 'SibSp'])
  # Plot
  hn = hn.plot(white_list=['Survived', 'SibSp'])
  # Heatmap
  hn = hn.heatmap(white_list=['Survived', 'SibSp'])


**White list example in plot**

.. code-block:: python

  # In this example we will keep only the node Survived and Age
  
  # d3graph
  hn = hn.d3graph(white_list=['Survived', 'Age', 'Pclass'])
  # Plot
  hn = hn.plot(white_list=['Survived', 'Age', 'Pclass'])
  # Heatmap
  hn = hn.heatmap(white_list=['Survived', 'Age', 'Pclass'])

