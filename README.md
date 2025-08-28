[![Python](https://img.shields.io/pypi/pyversions/hnet)](https://img.shields.io/pypi/pyversions/hnet)
[![Pypi](https://img.shields.io/pypi/v/hnet)](https://pypi.org/project/hnet/)
[![Docs](https://img.shields.io/badge/Sphinx-Docs-Green)](https://erdogant.github.io/hnet/)
[![LOC](https://sloc.xyz/github/erdogant/hnet/?category=code)](https://github.com/erdogant/hnet/)
[![Downloads](https://static.pepy.tech/personalized-badge/hnet?period=month&units=international_system&left_color=grey&right_color=brightgreen&left_text=PyPI%20downloads/month)](https://pepy.tech/project/hnet)
[![Downloads](https://static.pepy.tech/personalized-badge/hnet?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/hnet)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/hnet/blob/master/LICENSE)
[![Forks](https://img.shields.io/github/forks/erdogant/hnet.svg)](https://github.com/erdogant/hnet/network)
[![Issues](https://img.shields.io/github/issues/erdogant/hnet.svg)](https://github.com/erdogant/hnet/issues)
[![Project Status](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![DOI](https://zenodo.org/badge/231843440.svg)](https://zenodo.org/badge/latestdoi/231843440)
[![Medium](https://img.shields.io/badge/Medium-Blog-black)](https://erdogant.github.io/hnet/pages/html/Documentation.html#medium-blog)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://erdogant.github.io/hnet/pages/html/Documentation.html#colab-notebook)
[![Donate](https://img.shields.io/badge/Support%20this%20project-grey.svg?logo=github%20sponsors)](https://erdogant.github.io/hnet/pages/html/Documentation.html#)
<!---[![BuyMeCoffee](https://img.shields.io/badge/buymea-coffee-yellow.svg)](https://www.buymeacoffee.com/erdogant)-->
<!---[![Coffee](https://img.shields.io/badge/coffee-black-grey.svg)](https://erdogant.github.io/donate/?currency=USD&amount=5)-->

<div>
<a href="https://erdogant.github.io/hnet/"><img src="https://github.com/erdogant/hnet/blob/master/docs/figs/logo.png" width="150" align="left" /></a>
hnet is a Python package for probability density fitting of univariate distributions for random variables.
The hnet library can determine the best fit for over 90 theoretical distributions. The goodness-of-fit test is used to score for the best fit and after finding the best-fitted theoretical distribution, the loc, scale, and arg parameters are returned.
It can be used for parametric, non-parametric, and discrete distributions. ⭐️Star it if you like it⭐️
</div>

---

### Key Features

| Feature | Description | Docs | Medium | Gumroad+Podcast|
|---------|-------------|------|------|-------|
| **Association Learning** | Discover significant associations across variables using statistical inference. | [Link](https://erdogant.github.io/hnet/pages/html/Examples.html#titanic-dataset) | [Link](https://erdogant.medium.com/uncover-hidden-patterns-in-your-tabular-datasets-all-you-need-is-the-right-statistics-6de38f6a8aa7) | [Link](https://erdogant.gumroad.com/l/uncover-hidden-patterns-in-your-tabular-datasets-all-you-need-is-the-right-statistics-6de38f6a8aa7) |
| **Mixed Data Handling** | Works with continuous, discrete, categorical, and nested variables without heavy preprocessing. | [Link](https://erdogant.github.io/hnet/pages/html/index.html) | - | - |
| **Summarization** | Summarize complex networks into interpretable structures. | [Link](https://erdogant.github.io/hnet/pages/html/Use%20Cases.html#summarize-results) | - | - |
| **Feature Importance** | Rank variables by importance within associations. | [Link](https://erdogant.github.io/hnet/pages/html/Use%20Cases.html#feature-importance) | - | - |
| **Interactive Visualizations** | Explore results with dynamic dashboards and d3-based visualizations. | [Dashboard](https://erdogant.github.io/hnet/pages/html/Documentation.html#online-web-interface) | - | [Titanic Example](https://erdogant.github.io/docs/d3graph/titanic_example/index.html) |
| **Performance Evaluation** | Compare accuracy with Bayesian association learning and benchmarks. | [Link](https://erdogant.github.io/hnet/pages/html/index.html) | [Link](https://arxiv.org/abs/2005.04679) | - |
| **Interactive Dashboard** | No data leaves your machine. All computations are performed locally. | [Link](https://erdogant.github.io/hnet/pages/html/Documentation.html#online-web-interface) | - | - |


---

### Resources and Links
- **Example Notebooks:** [Examples](https://erdogant.github.io/hnet/pages/html/Documentation.html)
- **Medium Blogs** [Medium](https://erdogant.github.io/hnet/pages/html/Documentation.html#medium-blogs)
- **Gumroad Blogs with podcast:** [GumRoad](https://erdogant.github.io/hnet/pages/html/Documentation.html#gumroad-products-with-podcasts)
- **Documentation:** [Website](https://erdogant.github.io/hnet)
- **Bug Reports and Feature Requests:** [GitHub Issues](https://github.com/erdogant/hnet/issues)
- Article: [arXiv](https://arxiv.org/abs/2005.04679)
- Article: [PDF](https://arxiv.org/pdf/2005.04679)

---

### Background

* 	HNet stands for graphical Hypergeometric Networks, which is a method where associations across variables are tested for significance by statistical inference.
	The aim is to determine a network with significant associations that can shed light on the complex relationships across variables.
	Input datasets can range from generic dataframes to nested data structures with lists, missing values and enumerations.

* 	Real-world data often contain measurements with both continuous and discrete values.
	Despite the availability of many libraries, data sets with mixed data types require intensive pre-processing steps,
	and it remains a challenge to describe the relationships between variables.
	The data understanding phase is crucial to the data-mining process, however, without making any assumptions on the data,
	the search space is super-exponential in the number of variables. A thorough data understanding phase is therefore not common practice.

*	Graphical hypergeometric networks (``HNet``), a method to test associations across variables for significance using statistical inference. The aim is to determine a network using only the significant associations in order to shed 		light on the complex relationships across variables. HNet processes raw unstructured data sets and outputs a network that consists of (partially) directed or undirected edges between the nodes (i.e., variables). To evaluate the 		accuracy of HNet, we used well known data sets and generated data sets with known ground truth. In addition, the performance of HNet is compared to Bayesian association learning.

* 	HNet showed high accuracy and performance in the detection of node links. In the case of the Alarm data set we can demonstrate on average an MCC score of 0.33 + 0.0002 (*P*<1x10-6), whereas Bayesian association learning resulted in 	an average MCC score of 0.52 + 0.006 (*P*<1x10-11), and randomly assigning edges resulted in a MCC score of 0.004 + 0.0003 (*P*=0.49). HNet overcomes processes raw unstructured data sets, it allows analysis of mixed data types, it 		easily scales up in number of variables, and allows detailed examination of the detected associations.

<p align="left">
  <a href="https://erdogant.github.io/hnet/pages/html/index.html">
  <img src="https://github.com/erdogant/hnet/blob/master/docs/figs/fig1.png" width="600" />
  </a>
</p>

---

### Installation

##### Install hnet from PyPI
```bash
pip install hnet
```

##### Install from Github source
```bash
pip install git+https://github.com/erdogant/hnet
```  

##### Imort Library
```python
import hnet
print(hnet.__version__)

# Import library
from hnet import hnet
```

<hr>

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

### Contributors

<p align="left">
  <a href="https://github.com/erdogant/hnet/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=erdogant/hnet" />
  </a>
</p>

### Maintainer
* Erdogan Taskesen, github: [erdogant](https://github.com/erdogant)
* Contributions are welcome.
* Yes! This library is entirely **free** but it runs on coffee! :) Feel free to support with a <a href="https://erdogant.github.io/donate/?currency=USD&amount=5">Coffee</a>.

[![Buy me a coffee](https://img.buymeacoffee.com/button-api/?text=Buy+me+a+coffee&emoji=&slug=erdogant&button_colour=FFDD00&font_colour=000000&font_family=Cookie&outline_colour=000000&coffee_colour=ffffff)](https://www.buymeacoffee.com/erdogant)






