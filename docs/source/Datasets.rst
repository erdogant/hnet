.. _code_directive:

-------------------------------------

Datasets
'''''''''

HNet learns the structure from datasets with mixed datatypes and with unknown function. Input datasets can range from generic dataframes to nested data structures with lists, missing values and enumerations. 

We decided to use the DataFrame of ``pandas`` as the input data type for ``hnet``. 
The columns represent the variables or features containing continues/categorical values. The rows are the samples.

Below is the titanic dataset that can be the input for ``hnet`` in its current form. The steps of pre-processing of the dataset is explained in the pre-processing section.

.. table::

     +-----+------------+---------+-------+----+--------+-----+----------+
     |	   | PassengerId| Survived| Pclass| ...|    Fare|Cabin| Embarked |
     +-----+------------+---------+-------+----+--------+-----+----------+
     |	0  |           1|        0|      3| ...|  7.2500|  NaN|        S |
     +-----+------------+---------+-------+----+--------+-----+----------+
     |	1  |           2|        1|      1| ...| 71.2833|  C85|        C |
     +-----+------------+---------+-------+----+--------+-----+----------+
     |	2  |           3|        1|      3| ...|  7.9250|  NaN|        S |
     +-----+------------+---------+-------+----+--------+-----+----------+
     |	3  |           4|        1|      1| ...| 53.1000| C123|        S |
     +-----+------------+---------+-------+----+--------+-----+----------+
     |	4  |           5|        0|      3| ...|  8.0500|  NaN|        S |
     +-----+------------+---------+-------+----+--------+-----+----------+
     |	.. |         ...|      ...|    ...| ...|     ...|  ...|      ... |
     +-----+------------+---------+-------+----+--------+-----+----------+
     |	886|         887|        0|      2| ...| 13.0000|  NaN|        S |
     +-----+------------+---------+-------+----+--------+-----+----------+
     |	887|         888|        1|      1| ...| 30.0000|  B42|        S |
     +-----+------------+---------+-------+----+--------+-----+----------+
     |	888|         889|        0|      3| ...| 23.4500|  NaN|        S |
     +-----+------------+---------+-------+----+--------+-----+----------+
     |	889|         890|        1|      1| ...| 30.0000| C148|        C |
     +-----+------------+---------+-------+----+--------+-----+----------+
     |	890|         891|        0|      3| ...|  7.7500|  NaN|        Q |
     +-----+------------+---------+-------+----+--------+-----+----------+



Import example dataset
----------------------

Importing an example data set can be performed using :func:`hnet.import_example`. This function provides some example datasets such as **sprinkler**, **titanic**, **student**. 
The titanic dataset is depiced above and the spinkler below.

.. code:: python

    import hnet

    # import example
    df = hnet.import_example('sprinkler')

    # print DataFrame
    print(df)


.. table::

      +-----+------+----------+-----+-----------+
      |	    |Cloudy| Sprinkler| Rain| Wet_Grass |
      +-----+------+----------+-----+-----------+
      | 0   |     0|         0|    0|         0 |
      +-----+------+----------+-----+-----------+
      | 1   |     1|         0|    1|         1 |
      +-----+------+----------+-----+-----------+
      | 2   |     0|         1|    0|         1 |
      +-----+------+----------+-----+-----------+
      | 3   |     1|         1|    1|         1 |
      +-----+------+----------+-----+-----------+
      | 4   |     1|         1|    1|         1 |
      +-----+------+----------+-----+-----------+
      | ..  |   ...|       ...|  ...|       ... |
      +-----+------+----------+-----+-----------+
      | 995 |     1|         0|    1|         1 |
      +-----+------+----------+-----+-----------+
      | 996 |     1|         0|    1|         1 |
      +-----+------+----------+-----+-----------+
      | 997 |     1|         0|    1|         1 |
      +-----+------+----------+-----+-----------+
      | 998 |     0|         0|    0|         0 |
      +-----+------+----------+-----+-----------+
      | 999 |     0|         1|    1|         1 |
      +-----+------+----------+-----+-----------+


Example of the student dataset containing mixed datatypes:

.. code:: python

    # import example
    df = hnet.import_example('student')

    # print DataFrame
    print(df)


.. table::

     +-----------+----------+----------+------+------+----+-----+-----+---------+------+
     |		 | school_GP| school_MS| sex_F| sex_M| ...| G3_8| G3_9|    G1_18| G2_0 |
     +-----------+----------+----------+------+------+----+-----+-----+---------+------+
     |  school_GP|       0.0|       0.0|   0.0|   0.0| ...|  0.0|  0.0| 0.000000|  0.0 |
     +-----------+----------+----------+------+------+----+-----+-----+---------+------+
     |  school_MS|       0.0|       0.0|   0.0|   0.0| ...|  0.0|  0.0| 0.000000|  0.0 |
     +-----------+----------+----------+------+------+----+-----+-----+---------+------+
     |  sex_F    |       0.0|       0.0|   0.0|   0.0| ...|  0.0|  0.0| 0.000000|  0.0 |
     +-----------+----------+----------+------+------+----+-----+-----+---------+------+
     |  sex_M    |       0.0|       0.0|   0.0|   0.0| ...|  0.0|  0.0| 0.000000|  0.0 |
     +-----------+----------+----------+------+------+----+-----+-----+---------+------+
     |  age_19   |       0.0|       0.0|   0.0|   0.0| ...|  0.0|  0.0| 0.000000|  0.0 |
     +-----------+----------+----------+------+------+----+-----+-----+---------+------+
     |  ... 	 |       ...|       ...|   ...|   ...| ...|  ...| ... |  ...    |  ... |
     +-----------+----------+----------+------+------+----+-----+-----+---------+------+
     |  G3_18    |       0.0|       0.0|   0.0|   0.0| ...|  0.0|  0.0| 2.931461|  0.0 |
     +-----------+----------+----------+------+------+----+-----+-----+---------+------+
     |  G3_8     |       0.0|       0.0|   0.0|   0.0| ...|  0.0|  0.0| 0.000000|  0.0 |
     +-----------+----------+----------+------+------+----+-----+-----+---------+------+
     |  G3_9     |       0.0|       0.0|   0.0|   0.0| ...|  0.0|  0.0| 0.000000|  0.0 |
     +-----------+----------+----------+------+------+----+-----+-----+---------+------+
     |  G1_18    |       0.0|       0.0|   0.0|   0.0| ...|  0.0|  0.0| 0.000000|  0.0 |
     +-----------+----------+----------+------+------+----+-----+-----+---------+------+
     |  G2_0     |       0.0|       0.0|   0.0|   0.0| ...|  0.0|  0.0| 0.000000|  0.0 |
     +-----------+----------+----------+------+------+----+-----+-----+---------+------+
			     	               	      	   	       		 

Import from csv
----------------

Importing data from a csv file can be performed using ``pandas``:

.. code:: python

    import pandas as pd

    data = pd.read_csv('./pathname/to/file.csv')



Import from sklearn
-------------------

Various example datasets are also present in ``sklean``.
See below a demonstration how to import and use these in ``hnet``.
However, datasets should contain at least 1 catagorical value. Datasets containing only continues values should follow a different method, perhaps ``t-SNE``, ``SVD``, ``UMAP``.

.. code:: python

    # Import library
    from sklearn import datasets

    # Import pandas
    import pandas as pd

    X = datasets.load_boston()
    df = pd.DataFrame(data=X['data'], columns=X['feature_names'])

    X = datasets.load_diabetes()
    df = pd.DataFrame(data=X['data'], columns=X['feature_names'])

