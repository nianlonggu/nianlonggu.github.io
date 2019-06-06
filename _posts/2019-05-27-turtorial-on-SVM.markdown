---
layout:     post
title:      "An Introduction to Support Vector Machines (SVM): Dual problem solution using GDS and SMO"
subtitle:   " 支持向量机(SVM)概述：使用梯度下降和SMO求解对偶问题"
date:       2019-05-27 08:58:00
author:     "Gu"
header-img: "img/post-bg-2019-05-23.jpg"
catalog: true
tags:
    - 支持向量机
    - 机器学习
    - SVM
    - Machine Learning
---
> Just to clarify, these contents are mainly summarized from the course I took: "Fundamental of Big Data Analytics", taught by Prof. Mathar Rudolf. For for information please visit: [https://www.ti.rwth-aachen.de](https://www.ti.rwth-aachen.de)


Recall of the SVM primal problem and dual problem:<br>
**Primal Problem**
<center>
	$$
	\min_{\mathbf{w},b}\ \frac{1}{2}\|\mathbf{w}\|^2\\
	\begin{align}
	s.t.\ \ & y_i(\mathbf{w}^T\mathbf{x}_i+b)\geq 1,\ i=1,\dots,n
	\end{align}
	$$
</center>
**Dual Problem**
<center>
	$$
	\max_{\lambda}\ \sum_{i=1}^{n}\lambda_i - \frac{1}{2}\sum_{i,j}\lambda_i \lambda_j y_i y_j \mathbf{x}_i^T\mathbf{x}_j\\
	\begin{align}
	s.t.\ \ & \lambda_i \geq 0,\ i=1,\dots,n\\
	& \sum_{i=1}^{n}\lambda_i y_i = 0
	\end{align} 
	$$
</center>
The the last post we introduced how to apply Lagrangian duality to SVM and how to get the primal optimum once we get the dual optimum. In this post we mainly discuss how to solve the dual problem and get the dual optimum.

## Gradient Descent Algorithm for Dual Problem
To apply GDS to SVM, we need to reformulate the objective function of the dual problem. Our new objective function will be:
<center>
	$$
	\min_{\lambda}L(\lambda)=-\sum_{i=1}^{n}\lambda_i + \frac{1}{2}\sum_{i,j}\lambda_i \lambda_j y_i y_j \mathbf{x}_i^T\mathbf{x}_j + \frac{c}{2}(\sum_{i=1}^{n}\lambda_i y_i)^2 \\
	s.t. \ \ \lambda_i\geq 0
	$$
</center>
where $$c>0$$ is the weighting factor for the constraint $$\sum_{i=1}^{n}\lambda_i y_i = 0$$. For the constraint $$\lambda_i\geq 0$$, we can satisfy this constraint by clipping $$\lambda$$ into the region $$[0,\infty)$$ after each back propagation during gradient descent. We can compute the gradient:

<center>
$$
\begin{align}
\frac{\partial{L}}{\partial{\lambda_i}} &= -1 + \sum_{j=1}^{n}\lambda_j y_j \mathbf{x}_j^T(y_i\mathbf{x}_i) + {c}\sum_{j=1}^{n}\lambda_j y_j y_i 
\end{align}
$$
</center>
where
$$\mu(z)=\begin{cases}
    1, & \text{if}\ z\geq0\\
    0, & \text{otherwise}  
\end{cases}$$

We can also use the [denominator layout](https://en.wikipedia.org/wiki/Matrix_calculus) to express the gradient of $$L$$ with respect to the vector of $$\mathbf{\lambda}=(\lambda_1, \lambda_2,\dots,\lambda_n)^T$$. Let $$\mathbf{X}=(\mathbf{x}_ 1,\dots,\mathbf{x}_ n)^T\in R^{nxp}$$ and $$\mathbf{y}=(y_1,\dots, y_n)^T\in R_n$$. Then we have:
<center>
	$$
	\frac{\partial{L}}{\partial{\lambda}} = -  \mathbf{1}_n +( (\mathbf{y}\mathbf{1}_p^T)\circ\mathbf{X})\mathbf{X}^T(\mathbf{y}\circ\lambda)+ c(\lambda^T\mathbf{y})\mathbf{y}
	$$
</center>
where $$\circ$$ stands for **element-wise product** and $$\mu(-\lambda)=( \mu(-\lambda_1),\dots,\mu(-\lambda_n))^T$$

The update rule of $$\lambda$$ is:
<center>
	$$
	\lambda \leftarrow \lambda - \alpha \frac{\partial{L}}{\partial{\lambda}}
	$$
</center>
where $$\alpha$$ is the learning rate.



**Implementation and Experiments**

I implement the GDS algorithm to compute the dual optimum and use it to solve the original SVM optimization problem. The code is available in my github [SupportVectorMachine/gds-dual-svm.py](https://github.com/nianlonggu/SupportVectorMachine/blob/master/gds-dual-svm.py). The change of the hyperplane over iterations is shown in figure [Hyperplane Over Iteration](#hyperplane-over-iteration)
<a name="hyperplane-over-iteration"></a>
<img src="https://nianlonggu.github.io/img/2019-05-27-SVM/hyperplane-over-iteration.gif" width="400" hegiht="203" />
*<center>Hyperplane Over Iteration</center>*