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
In this dataset, each $$\mathbf{x}_ i$$ is an 2 dimensional vector. We need to use GDS to find the optimal separating hyperplane, which is a line in this case. The code is shown below:

```python

import numpy as np
import matplotlib.pyplot as plt
import os
def generate_folder(path):
	if not os.path.exists(path):
		os.makedirs(path)
	return path

def load_data(num_samples = 1000, seed = 1000):
	np.random.seed(seed)
	x1 = np.random.multivariate_normal( [1,1], [[1,-0.3],[-0.3,2]], num_samples  )
	x2 = np.random.multivariate_normal( [7,7], [[1,-0.3],[-0.3,2]], num_samples )
	y1 = np.ones([num_samples]) *-1
	y2 = np.ones([num_samples]) *1
	x = np.concatenate([x1,x2], axis =0)
	y = np.concatenate([y1,y2], axis =0)
	return x, y

def plot_results( x,y, w=None, b=None, title = "", img_save_path = None , show_img = True ):
	x_negative_1 =[]
	x_positive_1 =[]
	for ind in range(x.shape[0]):
		if y[ind] == -1:
			x_negative_1.append( x[ind] )
		else:
			x_positive_1.append( x[ind] ) 
	x_negative_1 = np.asarray(x_negative_1)
	x_positive_1 = np.asarray(x_positive_1) 
	plt.plot(x_negative_1[:,0], x_negative_1[:,1], "o", markerfacecolor='none')
	plt.plot(x_positive_1[:,0], x_positive_1[:,1], 's', markerfacecolor='none')
	if w is not None and b is not None:
		if w[1] != 0:
			## a1 a2 represent the first and second dimensions
			a1 = np.linspace( min(x[:,0]), max(x[:,0]), 1000 )
			a2 = -w[0]/w[1]*a1-b/w[1]
			a2_up_margin = -w[0]/w[1]*a1-b/w[1]+1/w[1]
			a2_down_margin = -w[0]/w[1]*a1-b/w[1]-1/w[1]
			plt.plot(a1,a2,"r-")
			plt.plot(a1,a2_up_margin, "r--")
			plt.plot(a1,a2_down_margin, "r--")
		else:
			a2 = np.linspace( min(x[:,1]), max(x[:,1]), 1000 )
			a1 = -w[1]/w[0]*a2-b/w[0]
			a1_up_margin = -w[1]/w[0]*a2-b/w[0] + 1/w[0]
			a1_down_margin = -w[1]/w[0]*a2-b/w[0] - 1/w[0]
			plt.plot(a1,a2,"r-")
			plt.plot(a1_up_margin, a2, "r--")
			plt.plot(a1_down_margin, a2, "r--")
	plt.xlabel("x")
	plt.ylabel("y")
	plt.xlim([min(x[:,0])-0.5, max(x[:,0])+0.5 ])
	plt.ylim([min(x[:,1])-0.5, max(x[:,1])+0.5 ])
	plt.legend(["y=-1","y=+1"])
	plt.title(title)
	if img_save_path is not None:
		plt.savefig( img_save_path )
	if show_img:
		plt.show()
	plt.close()

## input the (x,y) of training dataset, output the hyperplane parameters w and b
def svm(x,y, max_iter = 100000, lr = 0.1, batch_size = 2000, mylambda = 0.0001 ):
	# initialize w and b
	x_dim = x.shape[-1]
	w = np.random.normal(size=x_dim)
	b = np.random.normal()
	current_iter = 0
	# start training
	while True:
		batch_index = np.random.choice( x.shape[0], size=batch_size ,replace = False )
		x_batch, y_batch = x[batch_index], y[batch_index]
		# define the loss function:
		loss1 = mylambda /2 * np.linalg.norm( w )**2
		loss2 = np.mean( np.maximum(1- y_batch*(np.matmul( x_batch,w )+b), 0) )
		loss = loss1 + loss2 
		dLdw = mylambda*w -  np.mean( np.expand_dims( ((1 - y_batch*(np.matmul( x_batch,w )+b)) >0)*y_batch, axis =-1) * x_batch, axis = 0 , keepdims = False)
		dLdb = - np.mean(((1- y_batch*(np.matmul( x_batch,w )+b))> 0)*y_batch, axis =0, keepdims = False)
		w -= lr * dLdw
		b -= lr * dLdb
		current_iter+=1
		if current_iter % 1000 ==0:
			print("regularization loss: %f, hinge loss: %f, totoal loss: %f"%( loss1,loss2, loss ))
			plot_results(x,y, w,b, title= "iteration %d"%(current_iter), img_save_path=generate_folder("results/gds-svm/")+"results-iter%d.jpg"%(current_iter), show_img=False )
		if current_iter >= max_iter:
			break
	return w,b

x, y = load_data()
np.random.seed()
w,b =svm(x,y)
plot_results(x,y, w,b)

```