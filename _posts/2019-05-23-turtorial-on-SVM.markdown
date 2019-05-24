---
layout:     post
title:      "An Introduction to Support Vector Machines (SVM)"
subtitle:   " 支持向量机(SVM)概述：原理与实现"
date:       2019-05-23 17:00:00
author:     "Gu"
header-img: "img/post-bg-2019-05-23.jpg"
catalog: true
tags:
    - 支持向量机
    - 机器学习
    - SVM
    - Machine Learning
---

> “Hello, my first notebook.”
## Contents
[What is SVM?](#what-is-SVM)<br>
[SVM in Linearly Separable Case](#SVM-in-Linearly-Separable-Case)

<p id = "what-is-SVM"></p>

## What is SVM?
Support Vector Machine (SVM) is a method for classification (and possibly for regression). Here we mainly discuss the most common application: **binary classification problem**. 
> Given a training dataset with binary classes {+1, -1}, SVM means to find a separating hyperplane which can maximize the margin. 

Here the margin represents the minimum distance from points of both classes to the hyperplane. **Support vectors** represent the points which are closest to the hyperplane. These points determine the position and direction of the hyperplane, so they are called "support vectors".
 <br>
 <!-- <center><img src="https://nianlonggu.github.io/img/2019-05-23-SVM/what-is-svm.svg"> </center> -->
<!-- <center>![what is svm](https://nianlonggu.github.io/img/2019-05-23-SVM/what-is-svm.svg) </center> center here will make it not work -->
<center><a name="linear-svm"></a></center>
![what is svm](https://nlgu.top/img/2019-05-23-SVM/what-is-svm.svg)
*<center>Linear SVM</center>*

In the series of SVMs, following aspects will be discusses:<br>
1. The optimization problem of SVM in linearly separable case
2. Using gradient descent algorithm to solve the SVM optimization problem
3. Lagrangian dual optimization and Karush-Kuhn-Tucker (KKT) condition
4. Dual problem of SVM and analysis
5. Solve the dual problem using Sequential Minimal Optimization (SMO)
6. The optimization problem of SVM with penalty term 
7. Kernel SVM for nonlinear cases

<p id = "SVM-in-Linearly-Separable-Case"></p>

### SVM in Linearly Separable Case
The linearly separable case means that in the training dataset, points with two different classes can be linearly separated by a hyperplane $H:\\{\mathbf{x}|\mathbf{w}^{T}\mathbf{x}+b=0 \\}$. The goal of SVM is to find the optimal parameters of $\hat{\mathbf{w}}$ and $\hat{b}$ which maximize the margin. See figure [Linear-SVM](#linear-svm).

Given a training dataset $\\{(\mathbf{x}_i,\ y_i)\\}, i=1,\dots,n$, where $\mathbf{x}\in{R^p}$ and $y\in\\{-1,+1\\}$. If such a separating hyperplane exists as shown in figure [Linear-SVM](#linear-svm), there exists a positive value, $\gamma$, which satisfies that 
* for any $y_i=+1$, $\mathbf{w}^T\mathbf{x}_i+b\geq \gamma$
* for any $y_i=-1$, $\mathbf{w}^T\mathbf{x}_i+b\leq -\gamma$

Those two conditions can be summarized into one condition by considering $y_i$ and $\mathbf{x}_ {i}$ jointly:
* $y_i(\mathbf{w}^T\mathbf{x}_ i+b)\geq \gamma \ \ \ $ for any $i=1,\dots,n$, 

The equality holds for **marginal points** ((like the blue square, and red circles in figure [Linear-SVM](#linear-svm))). This condition ensures that the hyperplane can correctly classify two classes in the training dataset, yet we have another condition in SVM which is to maximize the margin.

**How is the margin is defined?** <br>
The margin is defined by the Euclidean distance between the closest points (marginal points) to the hyperplane.