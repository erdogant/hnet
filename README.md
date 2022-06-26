[![Python](https://img.shields.io/pypi/pyversions/hnet)](https://img.shields.io/pypi/pyversions/hnet)
[![PyPI Version](https://img.shields.io/pypi/v/hnet)](https://pypi.org/project/hnet/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/erdogant/hnet/blob/master/LICENSE)
[![Github Forks](https://img.shields.io/github/forks/erdogant/hnet.svg)](https://github.com/erdogant/hnet/network)
[![GitHub Open Issues](https://img.shields.io/github/issues/erdogant/hnet.svg)](https://github.com/erdogant/hnet/issues)
[![Project Status](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Downloads](https://pepy.tech/badge/hnet/month)](https://pepy.tech/project/hnet/)
[![Downloads](https://pepy.tech/badge/hnet)](https://pepy.tech/project/hnet)
[![Sphinx](https://img.shields.io/badge/Sphinx-Docs-Green)](https://erdogant.github.io/hnet/)
[![arXiv](https://img.shields.io/badge/arXiv-Docs-Green)](https://arxiv.org/abs/2005.04679)
[![Medium](https://img.shields.io/badge/Medium-Blog-green)](https://towardsdatascience.com/explore-and-understand-your-data-with-a-network-of-significant-associations-9a03cf79d254)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erdogant/hnet/blob/master/notebooks/hnet.ipynb)
[![DOI](https://zenodo.org/badge/226647104.svg)](https://zenodo.org/badge/latestdoi/226647104)
<!---[![Coffee](https://img.shields.io/badge/coffee-black-grey.svg)](https://erdogant.github.io/donate/?currency=USD&amount=5)-->


# HNET - Association ruled based networks using graphical Hypergeometric Networks.

**Star this repo if you like it! ⭐️**
## 


## Blog

[**Read more details and usage about HNet in this blog!**](https://erdogant.github.io/hnet/pages/html/Documentation.html#)

## Online Web-interface!

[**Use HNet using a web-interface**](https://erdogant.github.io/hnet/pages/html/Documentation.html#online-web-interface)





## Summary

HNet stands for graphical Hypergeometric Networks, which is a method where associations across variables are tested for significance by statistical inference.
The aim is to determine a network with significant associations that can shed light on the complex relationships across variables.
Input datasets can range from generic dataframes to nested data structures with lists, missing values and enumerations.

Real-world data often contain measurements with both continuous and discrete values.
Despite the availability of many libraries, data sets with mixed data types require intensive pre-processing steps,
and it remains a challenge to describe the relationships between variables.
The data understanding phase is crucial to the data-mining process, however, without making any assumptions on the data,
the search space is super-exponential in the number of variables. A thorough data understanding phase is therefore not common practice.

**Methods**

We propose graphical hypergeometric networks (``HNet``), a method to test associations across variables for significance using statistical inference. The aim is to determine a network using only the significant associations in order to shed light on the complex relationships across variables. HNet processes raw unstructured data sets and outputs a network that consists of (partially) directed or undirected edges between the nodes (i.e., variables). To evaluate the accuracy of HNet, we used well known data sets and generated data sets with known ground truth. In addition, the performance of HNet is compared to Bayesian association learning.

**Results**

We demonstrate that HNet showed high accuracy and performance in the detection of node links. In the case of the Alarm data set we can demonstrate on average an MCC score of 0.33 + 0.0002 (*P*<1x10-6), whereas Bayesian association learning resulted in an average MCC score of 0.52 + 0.006 (*P*<1x10-11), and randomly assigning edges resulted in a MCC score of 0.004 + 0.0003 (*P*=0.49). 

**Conclusions**

HNet overcomes processes raw unstructured data sets, it allows analysis of mixed data types, it easily scales up in number of variables, and allows detailed examination of the detected associations.

**Documentation**

* API Documentation: https://erdogant.github.io/hnet/
* Article: https://arxiv.org/abs/2005.04679

## Method overview

<p align="left">
  <a href="https://erdogant.github.io/hnet/pages/html/index.html">
  <img src="https://github.com/erdogant/hnet/blob/master/docs/figs/fig1.png" width="600" />
  </a>
</p>

## Installation
* Install hnet from PyPI (recommended).

```bash
pip install -U hnet
```
## Examples

- Simple example for the Titanic data set

```python
# Initialize hnet with default settings
from hnet import hnet
# Load example dataset
df = hnet.import_example('titanic')
# Print to screen
print(df)
```

	#      PassengerId  Survived  Pclass  ...     Fare Cabin  Embarked
	# 0              1         0       3  ...   7.2500   NaN         S
	# 1              2         1       1  ...  71.2833   C85         C
	# 2              3         1       3  ...   7.9250   NaN         S
	# 3              4         1       1  ...  53.1000  C123         S
	# 4              5         0       3  ...   8.0500   NaN         S
	# ..           ...       ...     ...  ...      ...   ...       ...
	# 886          887         0       2  ...  13.0000   NaN         S
	# 887          888         1       1  ...  30.0000   B42         S
	# 888          889         0       3  ...  23.4500   NaN         S
	# 889          890         1       1  ...  30.0000  C148         C
	# 890          891         0       3  ...   7.7500   NaN         Q

#


##### <a href="https://erdogant.github.io/docs/d3graph/titanic_example/index.html">Play with the interactive Titanic results.</a> 
<link rel="import" href="https://erdogant.github.io/docs/d3graph/titanic_example/index.html">

# 

##### [Example: Learn association learning on the titanic dataset](https://erdogant.github.io/hnet/pages/html/Examples.html#titanic-dataset)

<p align="left">
  <a href="https://erdogant.github.io/hnet/pages/html/Examples.html#titanic-dataset">
     <img src="https://github.com/erdogant/hnet/blob/master/docs/figs/fig4.png" width="900" />
  </a>
</p>


#

##### [Example: Summarize results](https://erdogant.github.io/hnet/pages/html/Use%20Cases.html#summarize-results)

Networks can become giant hairballs and heatmaps unreadable. You may want to see the general associations between the categories, instead of the label-associations.
With the summarize functionality, the results will be summarized towards categories.

<p align="left">
  <a href="https://erdogant.github.io/hnet/pages/html/Use%20Cases.html#summarize-results">
  <img src="https://github.com/erdogant/hnet/blob/master/docs/figs/other/titanic_summarize_static_heatmap.png" width="300" />
  <a href="https://erdogant.github.io/docs/d3heatmap/d3heatmap.html">
     <img src="https://github.com/erdogant/hnet/blob/master/docs/figs/other/titanic_summarize_dynamic_heatmap.png" width="400" />
  </a>
</p>

<p align="left">
  <a href="https://erdogant.github.io/hnet/pages/html/Examples.html#titanic-dataset">
  <img src="https://github.com/erdogant/hnet/blob/master/docs/figs/other/titanic_summarize_static_graph.png" width="400" />
  <img src="https://github.com/erdogant/hnet/blob/master/docs/figs/other/titanic_summarize_dynamic_graph.png" width="400" />
  </a>
</p>





#

##### [Example: Feature importance](https://erdogant.github.io/hnet/pages/html/Use%20Cases.html#feature-importance)

<p align="left">
  <a href="https://erdogant.github.io/hnet/pages/html/Use%20Cases.html#feature-importance">
  <img src="https://github.com/erdogant/hnet/blob/master/docs/figs/other/feat_imp_1.png" width="600" />
  <img src="https://github.com/erdogant/hnet/blob/master/docs/figs/other/feat_imp_2.png" width="600" />
  <img src="https://github.com/erdogant/hnet/blob/master/docs/figs/other/feat_imp_3.png" width="600" />
  </a>
</p>

#


#### Performance

<p align="left">
  <a href="https://erdogant.github.io/hnet/pages/html/index.html">
  <img src="https://github.com/erdogant/hnet/blob/master/docs/figs/fig3.png" width="600" />
  </a>
</p>



<hr>

### Contribute
* All kinds of contributions are welcome!

### Citation
Please cite ``HNet`` in your publications if this is useful for your research. See column right for citation information.

* [arXiv](https://arxiv.org/abs/2005.04679)
* [Article in pdf](https://arxiv.org/pdf/2005.04679)
* [Sphinx](https://erdogant.github.io/hnet)
* [Github](https://github.com/erdogant/hnet)

### Maintainer
* Erdogan Taskesen, github: [erdogant](https://github.com/erdogant)
* Contributions are welcome.
* If you wish to buy me a <a href="https://erdogant.github.io/donate/?currency=USD&amount=5">Coffee</a> for this work, it is very appreciated :)
