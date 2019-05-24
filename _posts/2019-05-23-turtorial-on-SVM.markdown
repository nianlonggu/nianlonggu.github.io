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

<!-- > “Yeah It's on. ” -->

## Contents
[What is SVM?](#what-is-SVM)

<p id = "what-is-SVM"></p>

## What is SVM?
Support Vector Machine (SVM) is a method for classification (and possibly for regression). Here we mainly discuss the most common application: **binary classification problem**. 
> Given a training dataset with binary classes {+1, -1}, SVM means to find a separating hyperplane which can maximize the margin. 

Here the margin represents the minimum distance from points of both classes to the hyperplane. **Support vectors** represent the points which are closest to the hyperplane. These points determine the position and direction of the hyperplane, so they are called "support vectors".
 <br>



### Linea
> Given a training dataset $\{({x}_i, y_i)\}, i=1,2,...,n$, where $x_i$ is a $p$ dimensional vector and 


de  