---
layout:     post
title:      "An Introduction to Expectation-Maximization (EM) Algorithm and Gaussian Mixture Model (GMM) for Clustering"
subtitle:   " EM 算法与高斯混合模型"
date:       2019-07-07 21:09:00
author:     "Gu"
header-img: "img/post-bg-2019-07-07.jpg"
catalog: true
tags:
    - EM Algorithm
    - GMM
    - Clustering
    - 聚类
    - Machine Learning
    - 机器学习
---

Expectation Maximization (EM) algorithm is a special case of MLE where the observations (data samples $$\mathbf{x}$$) are inherently related with some hidden variables ($$\mathbf{z}$$). First of all, we need to review the basics of MLE.
## Maximum Likelihood Estimation (MLE)
Let $$\{\mathbf{x}_ i\},\ i=1,\dots,n$$ be a set of independent and identically distributed observations, and $$\mathbf{\theta}$$ be the parameters of the data distribution which are unknown for us. The maximum likelihood estimation of the parameters $$\theta$$ is the parameters which can maximize the joint distribution $$p_\theta(\mathbf{x}_ 1,\dots,\mathbf{x}_ n)= \prod_{i=1}^{n}p_\theta(\mathbf{x}_ i)$$
<center>
	$$
	\hat{\theta}_\text{MLE} = \arg\ \max_{\theta}{\prod_{i=1}^{n}p_\theta(\mathbf{x}_ i)}
	$$
</center>
More commonly, we choose to maximize the joint log-likelihood:
<center>
	$$
	\hat{\theta}_\text{MLE} = \arg\ \max_{\theta}{\sum_{i=1}^{n}\log p_\theta(\mathbf{x}_ i)}
	$$
</center>

We use an example to illustrate how it works (referred from [EM算法详解-知乎](https://zhuanlan.zhihu.com/p/40991784)). 

Suppose that we have a coin A, the likelihood of a heads is $$\theta_A$$. We denote one observation $$\mathbf{x}_ i=\{ x_ {i,1},x_ {i,2},x_ {i,3},x_ {i,4},x_ {i,5}, \}$$ as tossing the coin A 5 times and record the heads (1) or tails (0) of each tossing. For example, $$\mathbf{x}_ i$$ can be 01001, 01110, 10010, ... etc. The likelihood of the observation $$\mathbf{x}_ i$$ is:
<center>
	$$
	p({\mathbf{x}_i}) = \prod_{j=1}^{5}\theta_A^{x_{i,j}}(1-\theta_A)^{1-x_{i,j}} = \theta_A^{\sum_{j=1}^{5}x_{i,j}}(1-\theta_A)^{\sum_{j=1}^{5}(1-x_{i,j})}
	$$
</center>
Therefore, the log likelihood of the joint distribution of $$n$$ observations is:
<center>
	$$
	l(\theta_A) = \sum_{i=1}^{n} \sum_{j=1}^{5}\Big( x_{i,j}\log{\theta_A} +(1-x_{i,j})\log{(1-\theta_A)} \Big)
	$$
</center>
The MLE of $$\theta_A$$ is
<center>
	$$
	\hat{\theta}_{A,\text{MLE}} = \arg \max_{\theta_A} l(\theta_A)
	$$
</center>
To get $$\hat{\theta}_{A,\text{MLE}}$$ we can solve the equation $$\frac{\partial{l(\theta_A)}}{\partial{\theta_A}}=0$$.
<center>
	$$
	\begin{align}
	\frac{\partial{l(\theta_A)}}{\partial{\theta_A}} &= \sum_{i=1}^{n}\sum_{j=1}^5 \Big(\frac{x_{i,j}}{\theta_A} + \frac{x_{i,j}-1}{1-\theta_A} \Big)\\
	&=  \frac{\sum_{i=1}^{n}\sum_{j=1}^5 x_{i,j}}{\theta_A} + \frac{\sum_{i=1}^{n}\sum_{j=1}^5 x_{i,j}-5n}{1-\theta_A} \\
	&= 0
	\end{align}
	$$
</center>
Therefore, we have
<center>
	$$
	\hat{\theta}_{A, \text{MLE}} = \frac{\sum_{i=1}^{n}\sum_{j=1}^5 x_{i,j}}{5n}
	$$
</center>
This is actually equivalent to compute the average value of all tossing results. 
For example, if we have 10 observations as below:

 | $$\mathbf{x}_1$$ | 01011 |$$\mathbf{x}_6$$ | 01110 |
 | $$\mathbf{x}_2$$ | 01111 |$$\mathbf{x}_7$$ | 01110 |
 | $$\mathbf{x}_3$$ | 11011 |$$\mathbf{x}_8$$ | 11011 |
 | $$\mathbf{x}_4$$ | 00011 |$$\mathbf{x}_9$$ | 00100 |
 | $$\mathbf{x}_5$$ | 01010 |$$\mathbf{x}_{10}$$ | 01001 |

The sum of all tossing is 28, and the total number of tossing is 50, so MLE of $$\theta_A$$ is $$\frac{28}{50}=\frac{14}{25}$$

**MLE with hidden variables**

Now things become more complicated. Suppose we have two coins: A and B. The likelihood of a heads of coin A and B are $$\theta_A$$ and $$\theta_B$$ respectively. We want to find the MLE of $$\theta_A, \theta_B$$ using $$n$$ observations $$\{\mathbf{x}_ i\},\ i=1,\dots,n$$. Each observation has the same form as above. The challenging part is that for each observation $$\mathbf{x}_ i$$, we don't know which coin it comes from. For example, $$n=10$$, the observation set is the same as the table above. In this case how to find the MLE of $$\theta_A$$ and $$\theta_B$$?

This is an simple example where our observation is closely related with some hidden (unknown) variables. In other words, the information of the data is incomplete. The Expectation-Maximization algorithm can be used to solve these problems.

## Expectation-Maximization (EM) Algorithm
Before introducing EM algorithm, we need to known an important inequality: Jensen-Shannon Inequality.

**Jensen-Shannon Inequality**

If a function $$f(\mathbf{X})$$ is strictly convex, where $$\mathbf{X}$$ is a random variable and the Hessian matrix $$H$$ is positive definite, we have
<center>
	$$
	E_{X}[f(\mathbf{X})] \geq f(E_X(\mathbf{X}))
	$$
</center>
The equality holds if and only if $$E_X [\mathbf{X}]= \mathbf{X}$$ with the probability 1 ($$\mathbf{X}$$ is a constant). Note that of $$f(\mathbf{X})$$ is strictly concave, the direction of the inequality needs to be reversed.

We can use an example to illustrate Jensen-Shannon inequality more intuitively (**not proof**). 

<!-- ![png](https://nianlonggu.github.io/img/2019-07-04-SVM/SVM-Tutorial_12_0.png) -->

