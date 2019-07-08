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

We can use an example to illustrate Jensen-Shannon inequality more intuitively (**not proof**). This example is referred from [Andrew Ng's lecture note on EM](http://cs229.stanford.edu/notes/).
![jensen-shannon inequality](https://nianlonggu.github.io/img/2019-07-07-EM/js-inequality.svg)
As shown in this figure, the random variable $$\mathbf{X}$$ has only two possible values: $$a$$ and $$b$$, each with the probability 0.5. Therefore, $$f(E[\mathbf{X}])= f(\frac{a+b}{2})$$ and $$E[f(\mathbf{X})]=\frac{f(a)+f(b)}{2}$$. According to the convexity of the function $$f(\mathbf{X})$$, we have $$E[f(\mathbf{X})]\geq f(E[\mathbf{X}])$$, and the equality holds if and only if $$a=b$$, which means $$E[\mathbf{X}]=\mathbf{X}=a$$.

Now we have a powerful tool, and we will use it to deduce the EM algorithm.

**EM algorithm**\\
Recall the MLE problem:
<center>
	$$
	\theta_\text{MLE} = \arg \max_{\theta} \sum_{i=1}^{n} \log{p_\theta(\mathbf{x}_i)}
	$$
</center>
If $$\mathbf{x}$$ is related with a latent variable $$\mathbf{z}$$, we write $$p_\theta$$ as the marginal likelihood of the joint distribution:
<center>
	$$
	p_\theta(\mathbf{x}_i) = \sum_{z} p_\theta(\mathbf{x}_i, \mathbf{z})
	$$
</center>
Now our log-likelihood function $$l(\theta)$$ becomes:
<center>
	$$
	l(\theta) = \sum_{i=1}^{n}\log \sum_{z} p_\theta(\mathbf{x}_i, \mathbf{z})
	$$
</center>
The expression of the joint distribution is not known to us. To solve this maximization problem we introduce a distribution $$Q(\mathbf{z})$$, and rewrite the log-likelihood function as:
<center>
	$$
	\begin{align}
	l(\theta) &= \sum_{i=1}^{n} \log \sum_{z} Q(z) \frac{p_\theta(\mathbf{x}_i,\mathbf{z}))}{Q(\mathbf{z})}\\
	&= \sum_{i=1}^{n} \log E_{z\in Q(z)} \left[ \frac{p_\theta(\mathbf{x}_i,\mathbf{z}))}{Q(\mathbf{z})}\right]
	\end{align}
	$$
</center>
We know that the function $$f(x)=\log(x)$$ is strictly concave, so according to the Jensen-Shannon inequality, we have the following inequality:
<center>
	$$
	l(\theta) \geq L(\theta, Q) =\sum_{i=1}^{n} E_{z \in Q(z)}\left[ \log{\frac{p_\theta(\mathbf{x}_i,\mathbf{z}))} {Q(\mathbf{z})}} \right]
	$$
</center>
The equality holds if and only if $$\frac{p_\theta(\mathbf{x}_i,\mathbf{z}))} {Q(\mathbf{z})}$$ is a constant (with respect to variable $$\mathbf{z}$$). To achieve this we can set $$Q(\mathbf{z})= \frac{p_\theta(\mathbf{x}_i,\mathbf{z})}{\sum_{z}p_\theta(\mathbf{x}_i,\mathbf{z})}= \frac{ p_\theta(\mathbf{x}_i,\mathbf{z}) }{p_\theta(\mathbf{x}_i)}= p_\theta(\mathbf{z}\vert \mathbf{x}_i) $$. This means $$Q(\mathbf{z})$$ is the posterior of $$\mathbf{z}$$ given $$\mathbf{x}_i$$.

People may ask: why do we try to find a proper $$Q(\mathbf{z})$$ to make the equality hold?
> Our initial goal is to find the MLE of $\theta$ w.r.t $l(\theta)$. However, the original expression of $l(\theta)$ is not explicit, so we take advantage of the Jensen-Shannon inequality, by selecting a proper distribution of $Q(\mathbf{z})$, to make $l(\theta)=L(\theta, Q)$. Then we can instead maximize $L(\theta, Q)$ w.r.t $\theta$ to find the MLE in an iterative way.

Now, we can summarize the EM algorithm:

1. Heuristically (or randomly) initialize parameters $$\theta$$
2. Repeat:\\
* Expectation (**E**) step:\\
Based on current parameter $$\theta$$ and $$\mathbf{x}_ i$$, set $$Q(\mathbf{z})=p_\theta(\mathbf{z}\vert \mathbf{x}_ i)$$.
* Maximization (**M**) step:\\
Update $$\theta \leftarrow \arg \max_{\theta} \sum_{i=1}^{n}\sum_z Q(\mathbf{z}) \log \frac{p_\theta(\mathbf{x}_ i, \mathbf{z})}{Q(\mathbf{z})} $$

$$\ \ \ \ \ \ \ \ $$Until: $$\theta$$ converges

In [Andrew Ng's lecture notes](http://cs229.stanford.edu/notes/), it is proven that can guarantee that $$l(\theta)$$ is steadily maximized. Suppose that after $$l-1$$ iterations we have the log-likelihood $$l(\theta_{l-1})$$. At the $$l^\text{th}$$ iteration, after the **E** step, we have $$L(\theta_{l-1}, Q_l)= l(\theta_{l-1})$$; After the **M** step, updated $$\theta_{l}$$ is selected such that $$L(\theta_l, Q_l)\geq L(\theta_{l-1}, Q_l)$$. Then at the $$l+1$$ iteration, by selecting $$Q_{l+1}$$ as the posterior of $$\mathbf{z}$$, we have $$l(\theta_l)=L(\theta_l, Q_{l+1})$$. Therefore, we have
<center>
	$$
	l(\theta_l) = L(\theta_l, Q_{l+1}) \geq L(\theta_l, Q_{l}) \geq L(\theta_{l-1}, Q_{l}) = l(\theta_{l-1})
	$$
</center>
So we have $$l(\theta_{l})\geq l(\theta_{l-1})$$. This guarantees that the overall log-likelihood can only keep increasing or stay unchanged, but not decrease.

This deduction shows that the EM algorithm is heading to the right direction. However, this direction may not be the ideal one. It is pretty obvious that if $$l(\theta)$$ is globally concave, EM algorithm  can always converge at the global optimum. If $$l(\theta)$$ is not globally concave, the property $$l(\theta_l)\geq l(\theta_{l-1})$$ will guarantee that EM algorithm will converge at some point (assume that $$l(\theta)$$ is not delta function), but the converge point may not be globally optimum. 

Moreover, the EM algorithm is sensitive to the initialization. Different initialization may results in pretty different converge points, as shown in the figure below. 
![EM initialization](https://nianlonggu.github.io/img/2019-07-07-EM/example-em-init.svg) 
As shown in this figure, if the initialization is at point $$A$$, then EM will converge at point $$C_A$$, while the EM will converge at point $$C_B$$ if initialization is $$B$$. Obviously $$C_B$$ is the global optimum and $$C_A$$ not.

So how to make EM algorithm less sensitive to initialization and be more likely to find the global optimum? One simple, straight-forward but effective way is to randomly initialize the parameters and rum EM algorithm multiple times, and choose the parameters with the largest converged log-likelihood (objective function).