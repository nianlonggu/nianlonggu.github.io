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