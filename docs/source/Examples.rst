.. _code_directive:

-------------------------------------

Examples
-----------------

Lets load some datasets using :func:`hnet.import_example` and demonstrate the usage of ``hnet`` in learning Associations.


Sprinkler dataset
'''''''''''''''''''''

A natural way to study the relation between nodes in a network is to analyse the presence or absence of node-links. The sprinkler data set contains four nodes and therefore ideal to demonstrate the working of ``hnet`` in inferring a network. Links between two nodes of a network can either be undirected or directed (directed edges are indicated with arrows). Notably, a directed edge does imply directionality between the two nodes whereas undirected does not.

.. code-block:: python
	
	from hnet import hnet
	hn = hnet()

	# Import example dataset
	df = hn.import_example('sprinkler')

	# Learn the relationships
	results = hn.association_learning(df)

	# Generate the interactive graph
	G = hn.d3graph()


**Network output**

.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/sprinkler_example/index.html" height="500px" width="1000px", frameBorder="0"></iframe>


Titanic dataset
'''''''''''''''''''''

The titanic data set contains a data structure that is often seen in real use cases (i.e., the presence of categorical, boolean, and continues variables per sample) which is therefore ideal to demonstrate the steps of ``hnet``, and the interpretability. The first step is the typing of the 12 input features, followed by one-hot encoding. This resulted in a total of 2634 one hot encoded features for which only 18 features had the minimum required of `y_min=10` samples.

.. code-block:: python
	
	from hnet import hnet
	hn = hnet()

	# Import example dataset
	df = hn.import_example('titanic')

	# Learn the relationships
	results = hn.association_learning(df)

	# Generate the interactive graph
	G = hn.d3graph()


**Output looks as following**

.. code-block:: python

	# [DTYPES] Auto detecting dtypes
	# [DTYPES] [PassengerId] > [force]->[num] [891]
	# [DTYPES] [Survived]    > [int]  ->[cat] [2]
	# [DTYPES] [Pclass]      > [int]  ->[cat] [3]
	# [DTYPES] [Name]        > [obj]  ->[cat] [891]
	# [DTYPES] [Sex]         > [obj]  ->[cat] [2]
	# [DTYPES] [Age]         > [float]->[num] [88]
	# [DTYPES] [SibSp]       > [int]  ->[cat] [7]
	# [DTYPES] [Parch]       > [int]  ->[cat] [7]
	# [DTYPES] [Ticket]      > [obj]  ->[cat] [681]
	# [DTYPES] [Fare]        > [float]->[num] [248]
	# [DTYPES] [Cabin]       > [obj]  ->[cat] [147]
	# [DTYPES] [Embarked]    > [obj]  ->[cat] [3]
	# [DTYPES] Setting dtypes in dataframe
	#
	# [DF2ONEHOT] Working on PassengerId
	# [DF2ONEHOT] Working on Survived.....[2]
	# [DF2ONEHOT] Working on Pclass.....[3]
	# [DF2ONEHOT] Working on Name.....[891]
	# [DF2ONEHOT] Working on Sex.....[2]
	# [DF2ONEHOT] Working on Age
	# [DF2ONEHOT] Working on SibSp.....[7]
	# [DF2ONEHOT] Working on Ticket.....[681]
	# [DF2ONEHOT] Working on Fare
	# [DF2ONEHOT] Working on Cabin.....[148]
	# [DF2ONEHOT] Working on Embarked.....[4]
	# [DF2ONEHOT] Total onehot features: 19
	#
	# [HNET] Association learning across [19] features.
	# [HNET] Multiple test correction using holm
	# [HNET] Dropping Age
	# [HNET] Dropping Fare


Exernal link: https://erdogant.github.io/docs/d3graph/titanic_example/index.html

.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/titanic_example/index.html" height="1000px" width="100%", frameBorder="0"></iframe>


.. code-block:: python
	
	# Generate the interactive graph color on cluster label
	G = hn.d3graph()


.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/titanic_example/titanic_d3graph_cluster.html" height="1000px" width="100%", frameBorder="0"></iframe>



.. code-block:: python

	from hnet import hnet

	hn = hnet()

	df = hn.import_example()

	results = hn.association_learning(df)

	G_static = hn.plot()

	G = hn.heatmap()

	G = hn.d3graph()


.. code-block:: python

    import hnet

	[scores, adjmat] = hnet.compare_networks(out['simmatP'], out['simmatP'], showfig=True)

	adjmat_undirected = hnet.to_undirected(out['simmatLogP'])



.. code-block:: python
	
	# Generate the interactive graph color on cluster label
	G = hn.d3graph(node_color='cluster')


.. raw:: html

   <iframe src="https://erdogant.github.io/docs/d3graph/titanic_example/titanic_d3graph_cluster.html" height="1000px" width="100%", frameBorder="0"></iframe>
 
Heatmap d3-javascript
''''''''''''''''''''''''''''''


.. code-block:: python
	
	# Generate the interactive heatmap
	G = hn.d3heatmap()


.. raw:: html

   <iframe src="https://erdogant.github.io/docs/titanic/d3heatmap/titanic_heatmap.html" height="1000px" width="100%", frameBorder="0"></iframe>


black and white listing
''''''''''''''''''''''''''''''

Input variables (column names) can be black or white listed in the model.

**White list example**

.. code-block:: python

  from hnet import hnet

  # White list the underneath variables
  hn = hnet(white_list=['Survived', 'Pclass', 'Age', 'SibSp'])
  
  # Load data
  df = hn.import_example('titanic')
  
  # Association learning
  out = hn.association_learning(df)

  # [hnet] >Association learning across [10] categories.
  # 100%|---------| 10/10 [00:01<00:00,  7.27it/s]
  # [hnet] >Total number of computations: [171]
  # [hnet] >Multiple test correction using holm
  # [hnet] >Dropping Age


**Black list example**

.. code-block:: python

  from hnet import hnet

  # Black list the underneath variables
  hn = hnet(black_list=['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp'])
  
  # Load data
  df = hn.import_example('titanic')
  
  # Association learning
  out = hn.association_learning(df)

  # [hnet] >Association learning across [7] categories.
  # 100%|---------| 7/7 [00:11<00:00,  1.62s/it]
  # [hnet] >Total number of computations: [1182]
  # [hnet] >Multiple test correction using holm
  # [hnet] >Dropping Fare




.. raw:: html

	<hr>
	<center>
		<script async type="text/javascript" src="//cdn.carbonads.com/carbon.js?serve=CEADP27U&placement=erdogantgithubio" id="_carbonads_js"></script>
	</center>
	<hr>

