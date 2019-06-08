---
layout:     post
title:      "An Introduction to Support Vector Machines (SVM): SVM with slack variables"
subtitle:   " 支持向量机(SVM)概述：具有松弛变量的支持向量机"
date:       2019-06-07 16:06:00
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


Recall of the SVM primal problem:<br>
<center>
	$$
	\min_{\mathbf{w},b}\ \frac{1}{2}\|\mathbf{w}\|^2\\
	\begin{align}
	s.t.\ \ & y_i(\mathbf{w}^T\mathbf{x}_i+b)\geq 1,\ i=1,\dots,n
	\end{align}
	$$
</center>

This is the primal problem of the SVM in the case where points of two classes are linearly separable. Such a primal problem has two drawbacks:
* The separating plane is sensitive to (easily influenced by) outliers.
* Not suitable for the case where points of two classes are not linearly separable.

1. **The separating plane is sensitive to (easily influenced by) outliers.**
<a name="hyperplane-influenced-by-outliers"></a>
<img src="https://nianlonggu.github.io/img/2019-06-07-SVM/svm-outlier.svg"/> 
<!-- width="400" hegiht="203" /> -->
*<center>Hyperplane Influenced by Outliers</center>*
Figure [Hyperplane Influenced by Outliers](#hyperplane-influenced-by-outliers) shows how a single outlier greatly influences the final results of the hyperplane. This is due to the constraints $$y_i(\mathbf{w}^T\mathbf{x}_ i+b)\geq 1$$ in the primal problem will make sure that the minimum geodesic distance between points and the separating hyperplane is $$\frac{1}{\|\mathbf{w}\|}$$. When there is an outlier, in order to satisfy the constraints, the model will choose a smaller $$\|\mathbf{w}\|$$ and also greatly change the rotation/position of the separating hyperplane. However, using the separating hyperplane in Figure (b) is not a good choice, since compared with (a), in (b) the points have a much smaller average geodesic distance to the separating hyperplane. Therefore, it is more likely that the SVM makes wrong decisions when classifying new points.
2. **Not suitable for the case where points of two classes are not linearly separable.**\\
If the points are not linearly separable, then the SVM primal problem doesn't have a optimal solution, since there doesn't exist a certain $$\mathbf{w}$$ and $$b$$ which satisfies the constraints $$y_i(\mathbf{w}^T\mathbf{x}_ i+b)\geq 1$$.

## SVM with Slack Variables
To solve the problems above, we need to introduce a slack variable to the original SVM primal problem. This means that we allow certain (outlier) points to be within the margin or even cross the separating hyperplane, but such cases would be penalized. Now the primal problem of the "Slack-SVM" will be:

**Primal Problem**
<center>
	$$
	\min_{\mathbf{w},b}\ \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{n}\xi _i \\
	\begin{align}
	s.t.\ \  y_i(\mathbf{w}^T\mathbf{x}_i+b) &\geq 1-\xi_i ,\ &i=1,\dots,n \\
	  \xi_i  &\geq 0,\ &i=1,\dots,n
	\end{align}
	$$
</center>
Here $$\xi_i$$ is the slack variable, and the positive $$C$$ is the weight for the penalty term. Suppose that for some point $$\mathbf{x}_i$$, it holds $$y_i(\mathbf{w}^T\mathbf{x}_i+b) = 1-\xi_i$$:
* if $$\xi_i=0$$, then $$\mathbf{x}_i$$ is exactly at the marginal hyperplane (the margin for short).
* if $$0<\xi_i\leq 1$$, then $$\mathbf{x}_i$$ is located within the margin, but the label of $$\mathbf{x}_i$$ is correctly classified.
* if $$\xi_i > 1$$, then $$\mathbf{x}_i$$ is located at the other side of the separating hyperplane, which means a miss-classification.
<a name="Different xi and Point Locations"></a>
<img src="https://nianlonggu.github.io/img/2019-06-07-SVM/svm-slack-variable.svg"/> 
<!-- width="400" hegiht="203" /> -->
*<center>Different $\xi$ and Point Locations</center>*

It is possible to use Gradient Descent algorithm to solve the primal problem. However, due to the slack variables, the constraints is much more complex than the case without slack variables. It is more difficult to define the loss function used for gradient descent. On the contrary, the Lagrangian dual problem of this primal problem still remains compact and solvable, and can be easily extended to kernel SVM. Therefore, in the next, we mainly discuss the deduction of the Lagrangian dual problem of the Slack SVM primal problem.

**Lagrangian Function**
<center>
$$L( \mathbf{w}, b, \mathbf{\xi}, \lambda, \mu )= \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^{n}\xi_i  + \sum_{i=1}^{n}\lambda_i ( 1-\xi_i - y_i(\mathbf{w}^T\mathbf{x}_ i+b ) ) - \sum_{i=1}^{n}\mu_i \xi_i  $$
</center>
**Lagrangian Dual function**
<center>
	$$
	g(\lambda, \mu) = \inf_{\mathbf{w}, b, \xi} L(\mathbf{w}, b, \xi, \lambda, \mu)
	$$
</center>
To get the dual function, we can compute the derivative and set them to 0.
<center>
	$$
	\frac{\partial{L}}{\partial{\mathbf{w}}} = \mathbf{w} - \sum_{i=1}^{n}\lambda_i y_i \mathbf{x}_i = 0 \\
	\frac{\partial{L}}{\partial{b}} = -\sum_{i=1}^{n}\lambda_i y_i = 0\\
	\frac{\partial{L}}{\partial{\xi_i}} = C - \lambda_i - \mu_i = 0
	$$
</center>
From these 3 equations we have
<center>
	$$
	\mathbf{w}^\star = \sum_{i=1}^{n} \lambda_i y_i \mathbf{x}_i\\
	\sum_{i=1}^{n} \lambda_i y_i = 0\\
	\mu_i = C-\lambda_i
	$$
</center>
Substitue them in the Lagrangian function, we can get the Lagrangian dual function:
<center>
	$$
	g(\lambda, \mu) = \frac{1}{2}\sum_{i,j}\lambda_i \lambda_j y_i y_j \mathbf{x}_i^T\mathbf{x}_j + C\sum_{i=1}^{n}\xi_i + \sum_{i=1}^{n}\lambda_i -\sum_{i=1}^{n}\lambda_i \xi_i \\-\sum_{i,j}\lambda_i \lambda_j y_i y_j \mathbf{x}_i^T\mathbf{x}_j - \sum_{i=1}^{n}\lambda_i y_i b - C\sum_{i=1}^{n}\xi_i + \sum_{i=1}^{n}\lambda_i \xi_i\\
	= \sum_{i=1}^{n}\lambda_i - \frac{1}{2}\sum_{i,j}\lambda_i \lambda_j y_i y_j \mathbf{x}_i^T\mathbf{x}_j
	$$
</center>
Therefore, the Lagrangian dual problem is:
<center>
	$$
	\max_{\lambda, \mu} \sum_{i=1}^{n}\lambda_i - \frac{1}{2}\sum_{i,j}\lambda_i \lambda_j y_i y_j \mathbf{x}_i^T\mathbf{x}_j\\
	\begin{align}
		s.t.\ & \lambda_i \geq 0\\
		 &\mu_i \geq 0 \\
		 &\mu_i = C-\lambda_i \\
		 &\sum_{i=1}^{n} \lambda_i y_i =0
	\end{align}
	$$
</center>
We can use $$\lambda_i$$ to represent $$\mu_i$$, and finally get the dual problem:

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
Compared with the dual problem for the SVM without slack variables, the only difference is that here the constraints of $$\lambda$$ are $$0\leq \lambda_i \leq C$$, instead of $$\lambda_i \geq 0$$. 

Actually in the primal problem of the SVM without slack variables, we can think there is a hidden $$C=\infty$$, which means that the penalty of slack variables is infinitely large, so all points need to satisfy $$y_i(\mathbf{w}^T\mathbf{x}_ i+b)\geq 1$$.

**Solution of the Dual Problem**
1. Gradient Descent Algorithm
The objective function for gradient descent is:<center>
	$$
	\min_{\lambda} L(\lambda) = -\sum_{i=1}^{n} \lambda_i + \frac{1}{2}\sum_{i,j}\lambda_i \lambda_j y_i y_j \mathbf{x}_ i^T \mathbf{x}_ j + \frac{c}{2}(\sum_{i=1}^{n}\lambda_i y_i)^2 \\
	s.t.\ 0\leq \lambda_i \leq C \ , \ i=1,\dots,n
	$$</center>
Compared with the post [An Introduction to Support Vector Machines (SVM): Dual problem solution using GD](https://nianlonggu.github.io/2019/05/27/turtorial-on-SVM/), the objective function is the same. The only difference is that here the constraints are $$0\leq \lambda_i \leq C$$. To achieve this constraints, we can clip the value of $$\lambda_i$$ in the range $$[0,C]$$ after each gradient descent back propagation. For the detail of the gradient form, please have a look at that post. 

2. Sequential Minimal Optimization (SMO), which will be discussed in the following posts.

**Discussion on the Karush-Kuhn-Tucker (KKT) conditions**
The KKT conditions are now slightly different, since now in the dual function there are actually two variables: $$\lambda$$ and $$\mu$$. For the primal optimum $$\mathbf{w}^\star, b^\star, \xi^\star$$ and the dual optimum $$\lambda^\star, \mu^\star $$, it holds:
1. primal constraints<center>
	$$
	y_i({\mathbf{w}^\star}^T\mathbf{x}_ i +b^\star) \geq 1-\xi^\star_i \\
	\xi^\star_i \geq 0
	$$</center>
2. compute the infimum of $$L$$ w.r.t $$\mathbf{w}$$ and $$\xi$$<center>
	$$
	\Delta_{\mathbf{w},b,\xi}L( \mathbf{w}^\star, b^\star, \xi^\star, \lambda^\star, \mu^\star)=0
	$$</center>
3. dual constraints<center>
	$$
	\lambda_i^\star \geq 0\\
	\mu_i^\star \geq 0\\
	<!-- \text{These two comes with the condition 2:}\\ -->
	\sum_{i=1}^{n}\lambda_i^\star y_i =0\\
	\mu_i^\star = C - \lambda_i^\star
	$$</center>
4. Complementary Slackness<center>
	$$
	\lambda_i^\star ( 1-\xi_i^\star - y_i({\mathbf{w}^\star}^T\mathbf{x}_ i  +b^\star ) ) =0 \\
	\mu_i^\star \xi_i^\star = 0
	$$</center>

The complementary slackness is interesting. Suppose that we have already find the primal optimum and dual optimum. We can analyze the location of the point $$\mathbf{x}_ i$$ based on the value of $$\lambda_i$$:
1. $$\lambda_i^\star=0$$\\
 then $$\mu_i^\star = C$$, so $$\xi^\star_i=0$$, and $$y_i({\mathbf{w}^\star}^T\mathbf{x}_ i+b^\star)\geq 1$$. This means the distance from point $$\mathbf{x}_ i$$ to the separating hyperplane is greater than or equal to $$\frac{1}{\|\mathbf{w}^\star\|}$$. This point $$\mathbf{x}_ i$$ is not support vector.
2. $$0<\lambda_i^\star<C$$\\
 then $$0<\mu_i^\star<C$$, so $$\xi_i^\star =0$$, and $$y_i({\mathbf{w}^\star}^T\mathbf{x}_ i+b^\star)=1$$. This means that $$\mathbf{x}_ i$$ is exactly located at the margin hyperplane: the distance to the separating hyperplane is exactly $$\frac{1}{\|\mathbf{w}^\star\|}$$. The point $$\mathbf{x}_ i$$ is a support vector which is used to compute $$\mathbf{w}^\star$$ and $$b^\star$$.
3. $$\lambda_i^\star = C$$\\
 then $$\mu_i^\star =0$$, so $$\xi^\star_i\geq 0$$, and $$y_i({\mathbf{w}^\star}^T\mathbf{x}_ i+b^\star)=1-\xi^\star_i$$. This means that $$\mathbf{x}_ i$$ is within the margin, or even located in the other side of the separating hyperplane (miss-classification). The point $$\mathbf{x}_ i$$ is also a support vector which is used to compute $$\mathbf{w}^\star$$, **but not used to compute $$b^\star$$**.

 
Suppose that we have solved the dual problem and get the dual optimum. Let $$S_w=\{ i \vert 0<\lambda_i^\star \leq C \}$$ represent the support set related with $$\mathbf{w}$$; $$S_b=\{ i \vert 0<\lambda_i^\star < C \}$$ represent the support set related with $$b$$. Meanwhile, we define $$S_b^+ =\{ i \vert i\in S_b \  \text{and}\ y_i = +1 \}$$ and $$S_b^-=\{ i \vert i\in S_b\ \text{and}\ y_i = -1 \}$$. Then we can compute the primal optimum:
<center>
	$$
	\mathbf{w}^\star = 
	$$
</center>