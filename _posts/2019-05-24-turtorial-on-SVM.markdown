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
<center>$$\text{min}_{\mathbf{w},b}\ L=\frac{\lambda}{2}\|\mathbf{w}\|^2+\frac{1}{n}\sum^{n}_{i=1}{\max\{1-y_i(\mathbf{w}^T\mathbf{x}_i+b) ,0\}}$$</center>
This objective function contains two terms. The first term is used to maximize the margin. This term is also called **regularization term**. The second term is a penalty term used to penalize the case where $$y_i(\mathbf{w}^T\mathbf{x}_i+b)<1$$, which represents incorrect/imperfect classification. Note that for the case $$y_i(\mathbf{w}^T\mathbf{x}_i+b)\geq 1$$ we don't need to penalize it, so we use a max function $$\max\{1-y_i(\mathbf{w}^T\mathbf{x}_i+b) ,0\}$$. This is also called **hinge loss**.
<center>
	$$
		h(z) = \max\{1-z, 0\}
	$$
</center>
![hinge function](https://nlgu.top/img/2019-05-24-SVM/hinge-func.svg)
*<center>It looks like a hinge, isn't it?</center>*

$$\lambda$$ is a weight parameter used to control the weight of the regularization term. If $$\lambda$$ is too small, the model (the learned hyperplane) will mainly focuses on correctly classify the training data, but the margin may not be maximized. If $$\lambda$$ is too large, the model will have have a large margin, while there may exist more miss-classified points in the training dataset.  

**Compute the gradient**<br>
To apply GDS we also need get the exact expression of the gradient.
<center> $$\frac{\partial{L}}{\partial{\mathbf{w}}}=\lambda\mathbf{w}-\frac{1}{n}\sum_{i=1}^{n}{u(1-y_i(\mathbf{w}^T\mathbf{x}_i+b))y_i\mathbf{x}_i} $$ </center>
<center>
	$$
	\frac{\partial{L}}{\partial{b}}=-\frac{1}{n}\sum_{i=1}^{n}{u(1-y_i(\mathbf{w}^T\mathbf{x}_i+b))y_i}
	$$
</center>
where
<center>$$u(z)=\begin{cases}
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

Note that in practice that in each update loop we may not use the whole training dataset, instead we may use a mini-batch. Suppose that the mini batch size is $$m$$, then the expression of the gradient is:
<center>
	$$
		\frac{\partial{L}}{\partial{\mathbf{w}}}=\lambda\mathbf{w}-\frac{1}{m}\sum_{i=1}^{m}{u(1-y_i(\mathbf{w}^T\mathbf{x}_i+b))y_i\mathbf{x}_i}\\
		\frac{\partial{L}}{\partial{b}}=-\frac{1}{m}\sum_{i=1}^{m}{u(1-y_i(\mathbf{w}^T\mathbf{x}_i+b))y_i}
	$$
</center>
In the following we will use this mini-batch style expression.

## Code Implementation
To test the GDS algorithm, we use toy data shown in figure [2d toy data](#2d-toy-data)
<a name="2d-toy-data"></a>
<img src="https://nianlonggu.github.io/img/2019-05-24-SVM/gds-svm-raw-data.svg" width="400" hegiht="203" />
*<center>2d toy data</center>*
In this dataset, each $$\mathbf{x}_ i$$ is an 2 dimensional vector. In total there are 2000 samples. We need to use GDS to find the optimal separating hyperplane, which is a line in this case. The code is available in my github: [SupportVectorMachine/gds-svm.py](https://github.com/nianlonggu/SupportVectorMachine).

## Experiment and Analysis
**Visualization of Hyperplane**<br>
In this part, we set $$\lambda=1e-4, \text{learning_rate}=0.1, \text{batch_size}=100, \text{maximum_iteration}=100000$$. The change of the hyperplane over iterations is shown in figure [Hyperplane Over Iteration](#hyperplane-over-iteration)
<a name="hyperplane-over-iteration"></a>
<img src="https://nianlonggu.github.io/img/2019-05-24-SVM/hyperplane-over-iteration.gif" width="400" hegiht="203" />
*<center>Hyperplane Over Iteration</center>*
After 100000 iterations the hyperplane looks accurate and the margin seems to be maximized. If we compare the final result with the SVM illustration figure, we will find that they are very similar, which implies that the gradient descent algorithm does work!
<a name="hyperplane-over-iteration"></a>
<img src="https://nianlonggu.github.io/img/2019-05-24-SVM/compare-gds-svm.svg"  />
*<center>Comparison between experiment results and model illustration</center>*
**Influence of $$\lambda$$ on the final results**<br>







