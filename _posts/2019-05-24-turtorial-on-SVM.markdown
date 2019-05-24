---
layout:     post
title:      "An Introduction to Support Vector Machines (SVM): Gradient Descent Solution"
subtitle:   " 支持向量机(SVM)概述：梯度下降法"
date:       2019-05-24 18:00:00
author:     "Gu"
header-img: "img/post-bg-2019-05-23.jpg"
catalog: true
tags:
    - 支持向量机
    - 机器学习
    - SVM
    - Machine Learning
---

<!-- > “Hello, my first notebook.” -->

<!-- ## Contents
[Gradient Descent Algorithm](#GDS)<br>
[Lagrangian Dual Problem](#Lagrangian-Dual) -->
In the last article, we discussed that the SVM optimization problem is:<br>
<center>$$\text{min}\frac{1}{2}\|\mathbf{w}\|^2,\ \ \text{s.t.}\ \ \ y_i(\mathbf{w}^T\mathbf{x}_i+b)\geq 1, \ \ i=1,\dots,n$$</center>
To solve this optimization problem, there are multiple ways. One way is to treat this problem as a standard optimization problem and use gradient descent algorithm to compute the optimal parameters. Another way is to formulate the Lagrangian dual problem of the primal problem, transferring original optimization problem into an easier problem. Here we mainly discuss the first method.


<p id = "GDS"></p>

## Gradient Descent Algorithm
To apply GDS, we need to design a new objective function which is differentiable. The new objective function is:
<center>$$\text{min}_{\mathbf{w},b}\ L=\frac{1}{2}\|\mathbf{w}\|^2+c\sum^{n}_{i=1}{\max\{1-y_i(\mathbf{w}^T\mathbf{x}_i+b) ,0\}}$$</center>
This objective function contains two terms. The first term is used to maximize the margin. The second term is a penalty term used to penalize the case where $$y_i(\mathbf{w}^T\mathbf{x}_i+b)<1$$, which represents incorrect/imperfect classification.

$$c$$ is a weight parameter used to control the weight of the penalty term. If $$c$$ is too large, the model (the learned hyperplane) will mainly focuses on correctly classify the training data, but the margin may not be maximized. If $$c$$ is too small, the model will have have a large margin, while there may exist more miss-classified points in the training dataset.  

**Compute the gradient**<br>
To apply GDS we also need get the exact expression of the gradient.
<center> $$\frac{\partial{L}}{\partial{\mathbf{w}}}=\mathbf{w}-c\sum_{i=1}^{n}{h(1-y_i(\mathbf{w}^T\mathbf{x}_i+b))y_i\mathbf{x}_i} $$ </center>
<center>
	$$
	\frac{\partial{L}}{\partial{b}}=-c\sum_{i=1}^{n}{h(1-y_i(\mathbf{w}^T\mathbf{x}_i+b))y_i}
	$$
</center>
where
<center>$$h(z)=\begin{cases}
    1, & \text{if $z>0$}.\\
    0, & \text{otherwise}.
  \end{cases}$$
</center>
The updating rules of the parameter $$\mathbf{w}$$ and $$b$$ are:
<center>
	$$
	\mathbf{w}\leftarrow \mathbf{w} - \alpha\frac{\partial{L}}{\partial{\mathbf{w}}}\\
	b\leftarrow b - \alpha\frac{\partial{L}}{\partial{b}}\\
	$$
</center>
where $$\alpha$$ is the learning rate.

## Code Implementation
To test the GDS algorithm, we use toy data shown in figure 
<a id="2d-toy-data"></a>
![2d toy data](https://nlgu.top/img/2019-05-24-SVM/gds-svm-raw-data.svg)
