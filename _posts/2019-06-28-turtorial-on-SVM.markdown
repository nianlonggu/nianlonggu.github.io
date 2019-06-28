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
	L(\lambda) =& \lambda_a -\lambda_a y_a y_b - \xi y_b - \frac{1}{2}\lambda_a^2 K_{a,a} -\frac{1}{2}(\lambda_a y_a + \xi)^2 K_{b,b} 
	\end{align}
	$$
</center>



> Ref:
1. [机器学习算法实践-SVM中的SMO算法- 知乎](https://zhuanlan.zhihu.com/p/29212107)