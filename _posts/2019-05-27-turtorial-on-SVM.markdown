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
 