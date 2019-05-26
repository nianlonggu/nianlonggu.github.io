---
layout:     post
title:      "An Introduction to Support Vector Machines (SVM): Convex Optimization and Lagrangian Duality Principle"
subtitle:   " 支持向量机(SVM)概述：凸优化与拉格朗日对偶问题"
date:       2019-05-25 19:43:00
author:     "Gu"
header-img: "img/post-bg-2019-05-23.jpg"
catalog: true
tags:
    - 支持向量机
    - 机器学习
    - SVM
    - Machine Learning
---

In the last article we have conquered how to use gradient descent algorithm to train a SVM. So,
> is this the end of the story? 

Not really. Although using GDS can solve the SVM optimization, GDS has some shortcomings:
* Gradient procedure is time consuming and the solution may be suboptimal.
* GDS method cannot explicitly identify support vectors (points) which determine the hyperplane.

To overcome these shortcomings, we can take advantage of the Lagrangian duality. First we convert original SVM optimization problem into a primal (convex) optimization problem, then we can get the Lagrangian dual problem. Luckily we can solve the dual problem based on KKT condition using more efficient methods. 

First of all, we need to briefly introduce Lagrangian duality and Karush-Kuhn-Tucker (KKT) condition.

## Lagrangian Duality Principle
**Primal Problem**<br>
A primal convex optimization problem has the following expression:
<center>
	$$
	\min_{\mathbf{x}}  f_0(\mathbf{x})\\
	s.t. \ \ f_i(\mathbf{x}) \leq 0, \ i=1,\dots,m \\
	\ \ \ \ \ \ \ h_j(\mathbf{x}) = 0, \ j=1,\dots,p
	$$
</center>
where $f_i(\mathbf{x}) _{(i=0,1,\dots,m)}$ are convex, and $h_j(\mathbf{x}) _{(j=1,\dots,p)}$ are linear (or affine).

We can get the Lagrangian function:
<center>
	$$
	L(\mathbf{x}, \mathbf{\lambda}, \mathbf{\mu}) = f_0(\mathbf{x}) + \sum_{i=1}^{m}\lambda_{i}f_i(\mathbf{x}) + \sum_{j=1}^{p}\mu_jh_j(\mathbf{x})
	$$
</center>
Since $f_i(\mathbf{x})$ are convex, and $h_j(\mathbf{x})$ are linear, $$L(\mathbf{x}, \mathbf{\lambda}, \mathbf{\mu})$$ is also convex w.r.t $$\mathbf{x}$$. Therefore, we can get the infimum of $$L(\mathbf{x}, \mathbf{\lambda}, \mathbf{\mu})$$, which is called the Lagrangian dual function:
<center>
	$$
	g(\mathbf{\lambda},\mathbf{\mu})= \inf_\mathbf{x} \ L(\mathbf{x},\mathbf{\lambda},\mathbf{\mu})
	$$
</center>

> The difference between minimum and infimum: <br>
* $$\min(S)$$ means the smallest element in set $$S$$;<br>
* $$inf(S)$$ means the largest value which is less than or equal to any element in $$S$$.<br>
* In the case where the minimum value is reachable, infimum = minimum. e.g. $$S=\{\text{all natural number}\}$$, then $$\inf(S) = \min(S) = 0$$
* In the case where the minimum is not reachable, infimum may still exist. e.g. $$S=\{f(x)\vert f(x)=1/x, x>0\}$$, $$\inf(S)=0$$

**Dual Problem**
Based on the dual function we can get the dual optimization problem:
<center>
	$$
		\max_{\mathbf{\lambda},\mathbf{\mu}}\ g(\mathbf{\lambda},\mathbf{\mu})\\
		s.t. \ \ \lambda_i \geq 0, \ \ i=1,\dots,m\\
		\small\text{and other constraints introduced by computing the dual function}
	$$
</center>

**Strong Duality and Slater's Condition**<br>
Let $f_0^\star(x)$ and $g^\star(\mathbf{\lambda},\mathbf{\mu})$ be the primal optimum and dual optimum respectively. 
**Weak duality** means that
$$
	 g^\star(\mathbf{\lambda},\mathbf{\mu}) \leq f_0^\star(x)
$$ 
The difference $$f_0^\star(x)-g^\star(\mathbf{\lambda},\mathbf{\mu})$$ is called **duality gap**.

Under certain circumstances, the duality gap can be 0, which means the **strong duality** holds. This condition is called **Slater's condition**:<br>
* Apart from the constraints in primal problem, Slater's condition requires that the constraints $$f_i(\mathbf{x}) _ {(i=1,\dots,m)} $$ are linear (or affine).

If Slater's condition is satisfied, strong duality holds, and furthermore for the optimal value $$\mathbf{x}^\star$$, $$\mathbf{\lambda}^\star$$ and $$\mathbf{\mu}^\star$$, the **Karush-Kuhn-Tucker (KKT)** conditions also holds.

**Karush-Kuhn-Tucker (KKT) Conditions**
KKT conditions contain four conditions:
* 