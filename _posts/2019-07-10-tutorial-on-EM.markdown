---
layout:     post
title:      "EM Algorithm and Gaussian Mixture Model for Clustering"
subtitle:   "EM算法与高斯混合模型"
date:       2019-07-10 15:23:00
author:     "Gu"
header-img: "img/post-bg-2019-07-07.jpg"
catalog: true
tags:
    - EM Algorithm
    - GMM
    - Clustering
    - Machine Learning
    - 机器学习
---
In the last post on [EM algorithm](https://nianlonggu.github.io/2019/07/07/tutorial-on-EM/), we introduced the deduction of the EM algorithm and use it to solve the MLE of the heads probability of two coins. In this post, we will apply EM algorithm to more practical and useful problem, the Gaussian Mixture Model (GMM), and discuss about using GMM for clustering.

First, let's recall the EM algorithm:
> Suppose that we have the observations $$\{\mathbf{x}^{(i)}\}, i=1,\dots,n$$. $$\mathbf{x}^{(i)}$$ is related with a hidden variable $$\mathbf{z}$$ which is unknown to us. The task is to find the MLE of $$\theta$$: <center>$$
	\theta_\text{MLE} = \arg \max_{\theta} \sum_{i=1}^{n}\log \sum_{\mathbf{z}} p_\theta(\mathbf{z}, \mathbf{x}^{(i)})	
	$$</center>
The EM algorithm works as follows:
1. Randomly initialize $$\theta$$, set the $$\mathbf{z}$$ prior $$p(\mathbf{z})$$ 
2. **Repeat:**\\
At the $$l^\text{th}$$ iteration:
* **E** step:\\
set $$Q_{l}^{(i)}(\mathbf{z})=p_{\theta_{l-1}}(\mathbf{z}\vert \mathbf{x}_ i)$$ for $$i=1,\dots,n$$
* **M** step:\\
update $$\theta_{l}=\arg \max_{\theta} \sum_{i=1}^{n}Q_{l}^{(i)}(\mathbf{z})\log \frac{p_{\theta_{l-1}}(\mathbf{z}, \mathbf{x}^{(i)})}{Q_{l}^{(i)}(\mathbf{z})}$$
* Update the prior $$p(\mathbf{z})$$ (optional)
\\
\\
**Until** $$\theta$$ converges.

Based on the experience on solving coin tossing problem using EM, we can further deform the EM algorithm:
1. In the E step, according to Bayes Theorem, we have $$Q_{l}^{(i)}(\mathbf{z})=p_{\theta_{l-1}}(\mathbf{z}\vert \mathbf{x}^{(i)})=\frac{p_{\theta_{l-1}}(\mathbf{x}^{(i)}\vert \mathbf{z})p(\mathbf{z})}{p_{\theta_{l-1}}(\mathbf{x}^{(i)})}=\frac{ p_{\theta_{l-1}}(\mathbf{x}^{(i)}\vert \mathbf{z})p(\mathbf{z}) }{ \sum_{\mathbf{z}}p_{\theta_{l-1}}(\mathbf{x}^{(i)}\vert \mathbf{z})p(\mathbf{z}) }$$.\\
Here $$p(\mathbf{z})$$ is the prior of the latent variable $$\mathbf{z}$$. Either we initialize it as fixed distribution, or we dynamically update it over each iteration. If **the number of the value of the variable $$\mathbf{z}$$ is finite and traversable**, we can directly compute the sum $$\sum_{\mathbf{z}}p_{\theta_{l-1}}(\mathbf{x}^{(i)}\vert \mathbf{z})p(\mathbf{z})$$, and then compute the posterior $$Q_{l}^{(i)}(\mathbf{z})$$. For example, in two coin tossing problem, $$\mathbf{z}$$ can only be coin A (1) or coin B (0).
2. In the M step, the objective function <center>$$\begin{align}
	L(\theta_{l-1}, Q_{l}^{(i)} ) &= \sum_{i=1}^{n}Q_{l}^{(i)}(\mathbf{z})\log \frac{p_{\theta_{l-1}}(\mathbf{z},\mathbf{x}^{(i)})}{Q_{l}^{(\mathbf{z})}}\\
	&= \sum_{i=1}^{n}Q_{l}^{(i)}(\mathbf{z})\log \frac{p_{\theta_{l-1}}(\mathbf{x}^{(i)}\vert \mathbf{z})p(\mathbf{z})}{Q_{l}^{(i)}(\mathbf{z})}\\
	&= \sum_{i=1}^{n}Q_{l}^{(i)}(\mathbf{z})\log p_{\theta_{l-1}}(\mathbf{x}^{(i)}\vert \mathbf{z}) + \sum_{i=1}^{n}Q_{l}^{(i)}(\mathbf{z})\log \frac{p(\mathbf{z})}{Q_{l}^{(i)}(\mathbf{z})}
\end{align}
$$</center>
Since $$Q_{l}^{(i)}$$ is computed in the E step, in the M step it is treated as something which is independent of $$\theta$$. Moreover, the prior $$p(\mathbf{z})$$ is also assumed to be independent of $$\theta$$. Therefore, the term $$\sum_{i=1}^{n}Q_{l}^{(i)}(\mathbf{z})\log \frac{p(\mathbf{z})}{Q_{l}^{(i)}(\mathbf{z})}$$ is irrelevant to $$\theta$$, and the equivalent updating rule of $$\theta$$ can be written as: <center>$$\theta_{l}= \arg \max_{\theta} \sum_{i=1}^{n}Q_{l}^{(i)}(\mathbf{z})\log p_{\theta_{l-1}}(\mathbf{x}^{(i)}\vert \mathbf{z}) $$</center> 

## Gaussian Mixture Model (GMM)
As indicated by its name, the GMM is a mixture (actually a linear combination) of multiple Gaussian distributions. The probability density function of a GMM is ($$\mathbf{x}\in R^p$$):
<center>
	$$
	\begin{align}
	p(\mathbf{x}; \phi, \mu, \Sigma) 
	&= \sum_{m=1}^{M}\phi_{m} N(\mathbf{x};\mu_m, \Sigma_m)\\
	&= \sum_{m=1}^{M}\phi_{m} \frac{1}{(2\pi)^{\frac{p}{2}}\vert\Sigma_m\vert^{\frac{1}{2}} } \exp\{-\frac{1}{2}{(\mathbf{x}-\mu_m)^T \Sigma_m^{-1} (\mathbf{x}-\mu_m)}\}
	\end{align}
	$$
</center>
where $$M$$ is the number of Gaussian models. $$\phi_m$$ is the weight factor of the Gaussian model $$N(\mu_m,\Sigma_m)$$. 

GMM is very suitable to be used to fit the dataset which contains multiple clusters, and each cluster has circular or elliptical shape. For example, the data distribution shown in the following figure can be modeled by GMM.
