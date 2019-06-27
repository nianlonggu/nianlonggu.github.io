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
	\max_{\lambda, \mu} \sum_{i=1}^{n}\lambda_i - \frac{1}{2}\sum_{i,j}\lambda_i \lambda_j y_i y_j K_{i,j} \\
	\begin{align}
		s.t.\ & 0 \leq \lambda_i \leq C \\
		 &\sum_{i=1}^{n} \lambda_i y_i =0
	\end{align}
	$$
</center>
 
We have introduced using gradient descent algorithm to solve the dual problem. However, the computation of the gradient has a high time complexity and thus would be a challenge for memory, especially when the training dataset is large.










> Ref:
1. [机器学习算法实践-SVM中的SMO算法- 知乎](https://zhuanlan.zhihu.com/p/29212107)