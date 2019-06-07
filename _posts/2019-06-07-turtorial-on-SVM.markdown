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

This is the primal problem of the SVM in the case where points of two classes are linearly separable. Such a primal problem has two drawbacks:
* The separating plane is sensitive to (easily influenced by) outliers.
* Not suitable for the case where points of two classes are not linearly separable.

1. **The separating plane is sensitive to (easily influenced by) outliers.**
<a name="hyperplane-influenced-by-outliers"></a>
<img src="https://nianlonggu.github.io/img/2019-06-07-SVM/svm-outlier.svg"/> 
<!-- width="400" hegiht="203" /> -->
*<center>Hyperplane Influenced by Outliers</center>*
Figure [Hyperplane Influenced by Outliers](#hyperplane-influenced-by-outliers) shows how a single outlier greatly influences the final results of the hyperplane. This is due to the constraints $$y_i(\mathbf{w}^T\mathbf{x}_ i+b)\geq 1$$ in the primal problem will make sure that the minimum geodesic distance between points and the separating hyperplane is $$\frac{1}{\|\mathbf{w}\|}$$. When there is an outlier, in order to satisfy the constraints, the model will choose a smaller $$\|\mathbf{w}\|$$ and also greatly change the rotation/position of the separating hyperplane. However, using the separating hyperplane in Figure (b) is not a good choice, since compared with (a), in (b) the points have a much smaller average geodesic distance to the separating hyperplane. Therefore, it is more likely that the SVM makes wrong decisions when classifying new points.