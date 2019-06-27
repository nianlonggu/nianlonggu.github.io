---
layout:     post
title:      "An Introduction to Support Vector Machines (SVM): kernel functions"
subtitle:   " 支持向量机(SVM)概述：核函数"
date:       2019-06-27 15:10:00
author:     "Gu"
header-img: "img/post-bg-2019-05-23.jpg"
catalog: true
tags:
    - 支持向量机
    - 机器学习
    - SVM
    - Machine Learning
---


Recall of the Slack SVM dual problem:<br>

**Dual Problem**
<center>
	$$
	\max_{\lambda, \mu} \sum_{i=1}^{n}\lambda_i - \frac{1}{2}\sum_{i,j}\lambda_i \lambda_j y_i y_j \mathbf{x}_i^T\mathbf{x}_j\\
	\begin{align}
		s.t.\ & 0 \leq \lambda_i \leq C \\
		 &\sum_{i=1}^{n} \lambda_i y_i =0
	\end{align}
	$$
</center>
 
Suppose that we have solved the dual problem and get the dual optimum. Let $$S_w=\{ i \vert 0<\lambda_i^\star \leq C \}$$ represent the support set related with $$\mathbf{w}$$; $$S_b=\{ i \vert 0<\lambda_i^\star < C \}$$ represent the support set related with $$b$$. Meanwhile, we define $$S_b^+ =\{ i \vert i\in S_b \  \text{and}\ y_i = +1 \}$$ and $$S_b^-=\{ i \vert i\in S_b\ \text{and}\ y_i = -1 \}$$. Then we can compute the primal optimum:
<center>
	$$
	\mathbf{w}^\star = \sum_{i\in S_w}\lambda_i^\star y_i \mathbf{x}_i\\
	b^\star= y_j - {\mathbf{w}^\star}^T\mathbf{x}_j = y_j - \sum_{i\in S_w}\lambda_i^\star y_i \mathbf{x}_i^T \mathbf{x}_j \ , \ j\in S_b\\
	$$
</center>
Given a new point $$\mathbf{x}$$, we can perform classification by computing:
<center>
	$$
	\begin{align}
	\hat{y} &= {\mathbf{w}^\star}^T \mathbf{x} + b^\star\\
			&=\sum_{i\in S_w} \lambda^\star_i y_i \mathbf{x}_i^T \mathbf{x} + b^\star\\
	\end{align}
	$$
</center>

According to the formulas above, we notice that in the dual problem, computation of $$\mathbf{w}^\star$$ and classification of new points, $$\mathbf{x}_ i^T\mathbf{x}_ j$$ always appears as a whole.

## SVM with kernel functions
**Mapping points to a higher dimensional space**

In some cases, if the points is not linearly separable in current space, they are possibly linearly separable if we map them into the higher dimension.



