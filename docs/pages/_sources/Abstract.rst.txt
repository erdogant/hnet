
Abstract
''''''''

Background
    Real-world data often contain measurements with both continuous and discrete values. Despite the availability of many libraries, data sets with mixed data types require intensive pre-processing steps, and it remains a challenge to describe the relationships between variables. The data understanding phase is an important step in the data-mining process, however, without making any assumptions on the data, the search space is super-exponential in the number of variables.

Methods
    We propose graphical hypergeometric networks (HNet), a method to test associations across variables for significance using statistical inference. The aim is to determine a network using only the significant associations in order to shed light on the complex relationships across variables. HNet processes raw unstructured data sets and outputs a network that consists of (partially) directed or undirected edges between the nodes (i.e., variables). To evaluate the accuracy of HNet, we used well known data sets and in addition generated data sets with known ground truth. The performance of HNet is compared to Bayesian structure learning.

Results
    We demonstrate that HNet showed high accuracy and performance in the detection of node links. In the case of the Alarm data set we can demonstrate on average an MCC score of 0.33 + 0.0002 (P<1x10-6), whereas Bayesian structure learning resulted in an average MCC score of 0.52 + 0.006 (P<1x10-11), and randomly assigning edges resulted in a MCC score of 0.004 + 0.0003 (P=0.49).

Conclusions
    HNet can process raw unstructured data sets, allows analysis of mixed data types, it easily scales up in number of variables, and allows detailed examination of the detected associations.


Schematic overview
'''''''''''''''''''

The schematic overview of our approach is as following:

.. _schematic_overview:

.. figure:: ../figs/fig1.png




.. include:: add_bottom.add