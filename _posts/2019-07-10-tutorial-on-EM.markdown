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
	L(\theta_{l-1}, Q_{l}^{(i)} ) &= \sum_{i=1}^{n}\sum_{\mathbf{z}}Q_{l}^{(i)}(\mathbf{z})\log \frac{p_{\theta_{l-1}}(\mathbf{z},\mathbf{x}^{(i)})}{Q_{l}^{(\mathbf{z})}}\\
	&= \sum_{i=1}^{n}\sum_{\mathbf{z}}Q_{l}^{(i)}(\mathbf{z})\log \frac{p_{\theta_{l-1}}(\mathbf{x}^{(i)}\vert \mathbf{z})p(\mathbf{z})}{Q_{l}^{(i)}(\mathbf{z})}\\
	&= \sum_{i=1}^{n}\sum_{\mathbf{z}}Q_{l}^{(i)}(\mathbf{z})\log p_{\theta_{l-1}}(\mathbf{x}^{(i)}\vert \mathbf{z}) + \sum_{i=1}^{n}\sum_{\mathbf{z}}Q_{l}^{(i)}(\mathbf{z})\log \frac{p(\mathbf{z})}{Q_{l}^{(i)}(\mathbf{z})}
\end{align}
$$</center>
Since $$Q_{l}^{(i)}$$ is computed in the E step, in the M step it is treated as something which is independent of $$\theta$$. Moreover, the prior $$p(\mathbf{z})$$ is also assumed to be independent of $$\theta$$. Therefore, the term $$\sum_{i=1}^{n}\sum_{\mathbf{z}}Q_{l}^{(i)}(\mathbf{z})\log \frac{p(\mathbf{z})}{Q_{l}^{(i)}(\mathbf{z})}$$ is irrelevant to $$\theta$$, and the equivalent updating rule of $$\theta$$ can be written as: <center>$$\theta_{l}= \arg \max_{\theta} \sum_{i=1}^{n}\sum_{\mathbf{z}}Q_{l}^{(i)}(\mathbf{z})\log p_{\theta_{l-1}}(\mathbf{x}^{(i)}\vert \mathbf{z}) $$</center> 

## Gaussian Mixture Model (GMM)
As indicated by its name, the GMM is a mixture (actually a linear combination) of multiple Gaussian distributions. The probability density function of a GMM is ($$\mathbf{x}\in R^p$$):
<center>
	$$
	\begin{align}
	p(\mathbf{x}; \phi, \mu, \Sigma) 
	&= \sum_{j=1}^{M}\phi_{j} N(\mathbf{x};\mu_j, \Sigma_j)\\
	&= \sum_{j=1}^{M}\phi_{j} \frac{1}{(2\pi)^{\frac{p}{2}}\vert\Sigma_j\vert^{\frac{1}{2}} } \exp\{-\frac{1}{2}{(\mathbf{x}-\mu_j)^T \Sigma_j^{-1} (\mathbf{x}-\mu_j)}\}
	\end{align}
	$$
</center>
where $$M$$ is the number of Gaussian models. $$\phi_j$$ is the weight factor of the Gaussian model $$N(\mu_j,\Sigma_j)$$. Moreover, we have the constraint: $$\sum_{j=1}^{M} \phi_j =1$$. 

GMM is very suitable to be used to fit the dataset which contains multiple clusters, and each cluster has circular or elliptical shape. For example, the data distribution shown in the following figure can be modeled by GMM.
<img src="https://nianlonggu.github.io/img/2019-07-10-EM/GMM-distribution.png" width="400" hegiht="203" />
Now the question is: **given a dataset with the distribution in the figure above, if we want to use GMM to model it, how to find the MLE of the parameters ($$\phi,\mu,\Sigma$$) of the Gaussian Mixture Model?**

The answer is: using EM algorithm!
## EM algorithm on GMM parameters estimation
Suppose that there are $$M$$ Gaussian models in the GMM, our latent variable $$\mathbf{z}$$ only has $$M$$ different values: $$\{\mathbf{z}^{(j)}=j| j=1,\dots,M\}$$. Moreover, $$\mathbf{x}^{(i)}\in R^p$$. The EM algorithm works as follows:

1. Initialization: \\
We normalize the raw data if necessary, randomly initialize $$\phi, \mu, \Sigma$$, and initialize the prior of $$\mathbf{z}$$ as $$p(\mathbf{z}^{(j)})=\frac{1}{M}$$
2. Repeat:\\
**To avoid an overcomplicated expression, we omit the footnote of iteration index $$l$$**
* **E** step:\\
$$\begin{align}
Q^{(i)}(\mathbf{z}^{(j)})&=\frac{ p( \mathbf{x}^{(i)}\vert \mathbf{z}^{(j)} ; \phi, \mu,\Sigma  ) p(\mathbf{z}^{(j)})}{ \sum_{k=1}^{M}p( \mathbf{x}^{(i)}\vert \mathbf{z}^{(k)} ; \phi, \mu,\Sigma  ) p(\mathbf{z}^{(k)})  } \\
&=\frac{\phi_{j} \frac{1}{(2\pi)^{\frac{p}{2}}\vert \Sigma_j \vert^{\frac{1}{2}} }\exp\{-\frac{1}{2}(\mathbf{x}^{(i)}-\mu_j)^T\Sigma_j^{-1}(\mathbf{x}^{(i)}-\mu_j) \} p(\mathbf{z}^{(j)}) }{ \sum_{k=1}^{M} \phi_{k} \frac{1}{(2\pi)^{\frac{p}{2}}\vert \Sigma_k \vert^{\frac{1}{2}} }\exp\{-\frac{1}{2}(\mathbf{x}^{(i)}-\mu_k)^T\Sigma_k^{-1}(\mathbf{x}^{(i)}-\mu_k) \} p(\mathbf{z}^{(k)}) }
\end{align}$$\\
For short we denote $$q_{i,j}=Q^{(i)}(\mathbf{z}^{(j)})$$
* **M** step:\\
The objective function is:\\
$$L= \sum_{i=1}^{n}\sum_{j=1}^{M} q_{i,j} \log \phi_{j} \frac{1}{(2\pi)^{\frac{p}{2}}\vert\Sigma_j\vert^{\frac{1}{2}} } \exp\{-\frac{1}{2}{(\mathbf{x}^{(i)}-\mu_j)^T \Sigma_j^{-1} (\mathbf{x}^{(i)}-\mu_j)}\} $$\\
**Update $$\phi$$**\\
**We cannot update $$\phi_k$$ by simply computing
$$\frac{\partial{L}}{\partial{\phi_k}}= 0 $$, because $$\phi$$ also has another constraint: $$\sum_{j=1}^{M}=1$$**, so we can treat this as a convex optimization problem:<center>$$
\min_{\phi} -\sum_{i=1}^{n}\sum_{j=1}^{M} q_{i,j} \log{\phi_j N(\mathbf{x}^{(i)}\vert \mu_j, \Sigma_j) }\\
\text{s.t.}\  \sum_{j=1}^{M} \phi_j = 1 
$$</center>
Since $$N(\mathbf{x}^{(i)}\vert \mu_j,\Sigma_j)$$ is not related with $$\phi$$, the equivalent optimization problem is <center>$$
	\min_{\phi} -\sum_{i=1}^{n}\sum_{j=1}^{M} q_{i,j} \log{\phi_j }\\
	\text{s.t.}\  \sum_{j=1}^{M} \phi_j = 1 
$$</center>
**$$\phi_j > 0$$ is not regarded as a constraint, since it is already included in the objective function $$\log \phi_j$$**\\
The Lagrangian function <center>$$L(\phi, \lambda)= -\sum_{i=1}^{n}\sum_{j=1}^{M} q_{i,j} \log{\phi_j } + \lambda( \sum_{j=1}^{M}\phi_j -1 ) $$</center>
By solving $$\frac{\partial{L(\phi,\lambda)}}{\partial{\phi_k}}=-\sum_{i=1}^{n}\frac{q_{i,k}}{\phi_k}+\lambda=0$$, we have the primal optimum $$\phi_k^\star=\frac{\sum_{i=1}^{n}q_{i,k}}{\lambda}$$. Substituting it to the Lagrangian function, we have the dual function:<center>$$d(\lambda)=-\sum_{i=1}^{n}\sum_{j=1}^{M}q_{i,j}\log\frac{\sum_{l=1}^{n}q_{l,j}}{\lambda}+\lambda(\sum_{j=1}^{M}\frac{\sum_{l=1}^{n}q_{l,j}}{\lambda} -1)$$</center>. Therefore, the dual problem is: <center>$$
\lambda^\star=\arg\max_{\lambda} d(\lambda)
$$</center><center>$$\frac{\partial{d(\lambda)}}{\partial{\lambda}}=\sum_{i=1}^{n}\sum_{j=1}^{M}\frac{q_{i,j}}{\lambda}-1=0$$</center>So we have the dual optimum $$\lambda^\star=\sum_{i=1}^{n}\sum_{j=1}^{M}q_{i,j}=\sum_{i=1}^{n}1=n$$ Therefore, we get the update rule of $$\phi_k$$: <center>$$
\phi_k^\text{new} = \frac{1}{n}\sum_{i=1}^{n}q_{i,k},\ k=1,\dots,M
$$</center>
This deduction uses the Lagrangian Duality Principle, for detail please see my post [An Introduction to Support Vector Machines (SVM): Convex Optimization and Lagrangian Duality Principle](https://nianlonggu.github.io/2019/05/25/tutorial-on-SVM/)\\
**Update $$\mu$$**\\
We compute the partial derivative:<center>$$\begin{align}
\frac{\partial{L}}{\partial{\mu_k}}&=\frac{\partial}{\partial{\mu_k}}\Big[\sum_{i=1}^{n}-\frac{1}{2}q_{i,k}(\mathbf{x}^{(i)}-\mu_k)^T\Sigma_k^{-1}(\mathbf{x}^{(i)}-\mu_k)\Big]\\
&= -\sum_{i=1}^{n}q_{i,k}\Sigma_k^{-1}(\mu_k - \mathbf{x}^{(i)})\\
&= -\Sigma_k^{-1}( \sum_{i=1}^{n}q_{i,k}\mu_k -\sum_{i=1}^{n}q_{i,k}\mathbf{x}^{(i)}  )\\
&=0
\end{align}$$</center>
Since $$\Sigma_k>0$$, the solution to the above equation is:<center>$$\mu_k^\text{new} = \frac{ \sum_{i=1}^{n}q_{i,k}\mathbf{x}^{(i)} }{ \sum_{i=1}^{n}q_{i,k} },\ k=1,\dots,M$$</center>
**Update $$\Sigma$$**\\
let $$\Lambda_k=\Sigma_k^{-1}$$, then the objective function is $$L= \sum_{i=1}^{n}\sum_{j=1}^{M} q_{i,j} \log \phi_{j} \frac{\vert\Lambda_j\vert^{\frac{1}{2}}}{(2\pi)^{\frac{p}{2}} } \exp\{-\frac{1}{2}{(\mathbf{x}^{(i)}-\mu_j)^T \Lambda_j (\mathbf{x}^{(i)}-\mu_j)}\} $$\\
we compute the optimal $$\Lambda^\star_k$$ by solving the equation:<center>
$$\begin{align}\frac{\partial{L}}{\partial{\Lambda_k}}&=\frac{\partial}{\partial{\Lambda_k}}\Big\{ \sum_{i=1}^{n} q_{i,k}\Big[\frac{1}{2}\log \vert\Lambda_k\vert - \frac{1}{2}(\mathbf{x}^{(i)}-\mu_k)^T\Lambda_k(\mathbf{x}^{(i)}-\mu_k)\Big] \Big\} \\
&=\frac{\partial}{\partial{\Lambda_k}}\Big\{ \sum_{i=1}^{n} q_{i,k}\Big[\frac{1}{2}\log \vert\Lambda_k\vert - \frac{1}{2}\text{tr}\left((\mathbf{x}^{(i)}-\mu_k)^T\Lambda_k(\mathbf{x}^{(i)}-\mu_k)\right) \Big] \Big\} \\
&= \frac{1}{2}\sum_{i=1}^{n} q_{i,k} \Lambda_k^{-1} - \frac{1}{2} \frac{\partial}{\partial{\Lambda_k}} \text{tr}\left(\Lambda_k\sum_{i=1}^{n}q_{i,k}(\mathbf{x}^{(i)}-\mu_k)(\mathbf{x}^{(i)}-\mu_k)^T\right) \\
&= \frac{1}{2}\sum_{i=1}^{n} q_{i,k} \Sigma_k - \frac{1}{2} \sum_{i=1}^{n}q_{i,k}(\mathbf{x}^{(i)}-\mu_k)(\mathbf{x}^{(i)}-\mu_k)^T \\
&=0\end{align}$$</center>So we have the update rule of $$\Sigma_k$$:<center>$$
\Sigma_k^\text{new} = \frac{ \sum_{i=1}^{n}q_{i,k}(\mathbf{x}^{(i)}-\mu_k)(\mathbf{x}^{(i)}-\mu_k)^T }{\sum_{i=1}^{n} q_{i,k} }
$$</center>
* (optional) update the prior $$p(\mathbf{z}^{(j)})=\frac{1}{n}\sum_{i=1}^{n}Q^{(i)}(\mathbf{z}^{(j)})$$

$$\ \ \ \ \$$ Until all the parameters converges.