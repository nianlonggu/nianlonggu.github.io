---
layout:     post
title:      "An Introduction to Support Vector Machines (SVM): Sequential Minimal Optimization (SMO)"
subtitle:   " 支持向量机(SVM)概述：SMO算法求解对偶问题"
date:       2019-06-28 00:00:00
author:     "Gu"
header-img: "img/post-bg-2019-05-23.jpg"
catalog: true
tags:
    - 支持向量机
    - 机器学习
    - SVM
    - Machine Learning
---

Recall the Kernel SVM dual problem:


**Dual Problem**
<center>
	$$
	\max_{\lambda, \mu} L(\lambda)= \sum_{i=1}^{n}\lambda_i - \frac{1}{2}\sum_{i,j}\lambda_i \lambda_j y_i y_j K_{i,j} \\
	\begin{align}
		s.t.\ & 0 \leq \lambda_i \leq C \\
		 &\sum_{i=1}^{n} \lambda_i y_i =0
	\end{align}
	$$
</center>
 
We have introduced using gradient descent algorithm to solve the dual problem. However, the computation of the gradient has a high time complexity and thus would be a challenge for memory, especially when the training dataset is large. In this post, I introduce an efficient and light-version algorithm to solve the dual problem: Sequential Minimal Optimization (SMO)


## Sequential Minimal Optimization (SMO)
The algorithm of SMO is:
>
Initialization: let $$\{\lambda_i\}, i=1,\dots,n$$ be a set which satisfies the dual constraint.\\
Repeat:\\
$$\ \ \ \ $$(1) heuristically select two $$\lambda_a, \lambda_b$$, and set all the other $$\lambda_i (i\neq a,b)$$ fixed;\\
$$\ \ \ \ $$(2) optimize $$L(\lambda)$$ with respect to $$\lambda_a, \lambda_b$$;\\
Until: KKT condition is satisfied with certain accuracy.


First question about the initialization: how to find a set $$\{\lambda_i\}$$ which satisfies the dual constraints? \\
The answer is simply set $$\lambda_i=0$$ for $$i=0,\dots,n$$.

Suppose that we have finished the initialization, and pick up a pair $$\lambda_a, \lambda_b$$ to optimize while keeping $$\lambda_i (i\neq a,b)$$ fixed, then we have
<center>
	$$
	\begin{align}
	L(\lambda) =& \lambda_a + \lambda_b -\frac{1}{2} \lambda_a^2 K_{a,a} - \frac{1}{2} \lambda_b^2 K_{b,b} - \lambda_a \lambda_b y_a y_b K_{a,b} \\
	& -  \sum_{i\neq a,b} \lambda_a \lambda_i y_a y_i K_{a,i} -  \sum_{i \neq a,b} \lambda_b \lambda_i y_b y_i K_{b,i} + Const
	\end{align}
	$$
</center>
Moreover, according to the dual constraints, we have
<center>
$$
	\lambda_a y_a + \lambda_b y_b = -\sum_{i\neq a,b} \lambda_i y_i = - \xi\\
	\lambda_b y_b = -\lambda_a y_a -\xi\\
	\lambda_b = -\lambda_a y_a y_b -\xi y_b
$$	
</center>
So we have 
<center>
	$$
	\begin{align}
	L(\lambda) =& \lambda_a -\lambda_a y_a y_b - \xi y_b - \frac{1}{2}\lambda_a^2 K_{a,a} -\frac{1}{2}(\lambda_a y_a + \xi)^2 K_{b,b} \\
	& + \lambda_a y_a ( \lambda_a y_a + \xi ) K_{a,b} - \sum_{i\neq a,b} \lambda_a y_a \lambda_i y_i K_{a,i}\\
	& + \sum_{i\neq a,b}(\lambda_a y_a + \xi)\lambda_i y_i K_{b,i} + Const
	\end{align}
	$$
</center>
$$L(\lambda)$$ is concave with respect to $$\lambda_a$$, since $$\frac{\partial^2{L}}{\partial{\lambda_a^2}}= -( K_{a,a} + K_{b,b} - 2K_{a,b} )=-(e_a - e_b)^T \mathbf{K} (e_a - e_b) \leq 0 $$ due to the fact that the kernel matrix $$\mathbf{K}$$ is nonnegative definite (see last post [An Introduction to Support Vector Machines (SVM): kernel functions](http://nianlonggu.github.io/2019/06/27/turtorial-on-SVM/) ). Therefore, we can find the optimal value of $$\lambda_a$$ which maximizes $$L(\lambda)$$ by computing the gradient and set it to 0.
<center>
	$$
	\begin{align}
	\frac{\partial{L(\lambda)}}{\partial{\lambda_a}} =& 1 - y_a y_b -\lambda_a K_{a,a} - (\lambda_a y_a +\xi)y_a K_{b,b} + 2\lambda_a K_{a,b} \\
	&+ y_a \xi K_{a,b} - \sum_{i\neq a,b} y_a \lambda_i y_i K_{a,i} + \sum_{i \neq a,b}y_a \lambda_i y_i K_{b,i}\\
	=& 0
	\end{align}
	$$
</center>
By solving this equation, we will get the solution for $$\lambda_a^\star$$:
<center>
	$$
	\lambda_a^{\text{new}} = \frac{ 1-y_a y_b - \xi y_a K_{b,b} + y_a \xi K_{a,b} - \sum_{i \neq a,b} y_a \lambda_i y_i K_{a,i} +\sum_{i\neq a,b}y_a \lambda_i y_i K_{b,i} }{ K_{a,a} + K_{b,b} -2K_{a,b} }
	$$
</center>
It is too complicated to compute the numerator since there are too many terms. In the next, we will show that we can actually compute $$\lambda_a^\text{new}, \lambda_b^\text{new}$$ from the old $$\lambda_a^\text{old}, \lambda_b^\text{old} $$.

Before updating the value of $$\lambda_a, \lambda_b$$, we first use the old version $$\lambda$$ to perform the classification on data $$\mathbf{x}_ a, \mathbf{x}_ b$$:
<center>
	$$
	\begin{align}
	\hat{y}_a &= \sum_{i\neq a,b}\lambda_i y_i K_{i,a} + \lambda_a^\text{old} y_a K_{a,a} + \lambda_b^\text{old} y_b K_{b,a}\\
	\hat{y}_b &= \sum_{i\neq a,b}\lambda_i y_i K_{i,b} + \lambda_a^\text{old} y_a K_{a,b} + \lambda_b^\text{old} y_b K_{b,b}\\
	\end{align}
	$$
</center>
Base on the expressions of $$\hat{y}_a, \hat{y}_b$$, we can have the following equation:
<center>
	$$
	\begin{align}
	&y_a[ (\hat{y}_b - y_b) - (\hat{y}_a - y_a)  ]\\
	= & \sum_{i\neq a,b}y_a \lambda_i y_i K_{i,b} + \lambda_a^\text{old} K_{a,b} + \lambda_b^\text{old} y_a y_b K_{b,b} + y_a b^\text{old} - y_a y_b \\
	\ & - \sum_{i \neq a,b}y_a \lambda_i y_i K_{i,a} - \lambda_a^\text{old}K_{a,a} - \lambda_b^\text{old} y_a y_b K_{b,a} - y_a b^\text{old} +1\\
	=&  \sum_{i\neq a,b} y_a \lambda_i y_i K_{i,b} + \lambda_a^\text{old} K_{a,b} - \xi y_a K_{b,b}    - \lambda_a^\text{old} K_{b,b}- y_a y_b \\
	\ & - \sum_{i \neq a,b}y_a \lambda_i y_i K_{i,a} - \lambda_a^\text{old}K_{a,a}  + \lambda_a^\text{old} K_{a,b} + \xi y_a K_{a,b}   +1 \\
	=& 1- y_a y_b - \xi y_a K_{b,b} + \xi y_a K_{a,b} - \sum_{i \neq a,b}y_a \lambda_i y_i K_{i,a} +\sum_{i\neq a,b} y_a \lambda_i y_i K_{i,b}\\
	\ & -\lambda_a^\text{old}( K_{a,a} + K_{b,b} - 2K_{a,b} )\\
	= & \lambda_a^{\text{new}}(K_{a,a} + K_{b,b} -2K_{a,b})-\lambda_a^\text{old}( K_{a,a} + K_{b,b} - 2K_{a,b} )
	\end{align}
	$$
</center>
We denote prediction error $$E_i= \hat{y}_i - y_i$$, then we have the expression of $$\lambda_a^\text{new}$$:
<center>
	$$
	\lambda_a^\text{new} = \lambda_a^\text{old} + \frac{y_a(E_b - E_a)}{K_{a,a} +K_{b,b} - 2K_{a,b}  }
	$$
</center>

> Discussion: What if $$K_{a,a} +K_{b,b} - 2K_{a,b}=0$$? In this case $$L(\lambda)$$ is a first degree function, it's still concave, but in this case the definition of $$\lambda_a^\text{new}$$ is no longer meaningful, so we just simply select another pair $$(\lambda_a, \lambda_b)$$ and do the computation above.

Note that the expression of the $$\lambda_a^\text{new}$$ is not clipped, so for simplicity we name it as $$\lambda_a^\text{new, unclipped}$$. It is inadequate to only compute the $$\lambda_a^\text{new, unclipped}$$. We need to further clip it based on the meaningful domain determined by the dual constraints. According to the dual constraints, each $$\lambda_i$$ actually has a box constraint. So we have:
<center>
	$$
	0\leq \lambda_a \leq C\\
	0\leq \lambda_b \leq C\\
	\lambda_b = -\lambda_a y_a y_b - \xi y_b
	$$
</center>
We know that $$y_i \in {-1, +1}$$. Based on whether $$y_a = y_b$$ or not, we can have the relationship between $$\lambda_a$$ and $$\lambda_b$$ with box constraints, shown in the figure below.

<a name="lambda_ab"></a>
<img src="https://nianlonggu.github.io/img/2019-06-28-SVM/lambda_ab.svg"/>
*<center>Relationship between $\lambda_a$ and $\lambda_b$ with box constraints.</center>*

According to the figure, we can get the lower bound $$L$$ and higher bound $$H$$ for a meaningful solution of a new $$\lambda_a$$:
1. If $$y_a = y_b$$

> Ref:
1. [机器学习算法实践-SVM中的SMO算法- 知乎](https://zhuanlan.zhihu.com/p/29212107)


3 ACL
