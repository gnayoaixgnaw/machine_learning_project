# Machine Learning

***Explained machine learning models and projects based on ML***

Basic machine learning algorithms:

1.**Linear regression**

2.**Logistic regression**

3.**Support Vector Machines**

4.**Naive Bayes**

*(below coming soon...)*

5.Decision trees

6.Random Forest

7.k- Means clustering

8.k-Nearest neighbors



## Linear regression

Linear regression is very simple and basic.First, linear regression is supervised model, which means data should be labelled.Then, for linear regression, it will find the relationships between features(x1,x2,x3....), which represent as coefficients of these variables.


### simple linear regression


Let's look at a simple linear regression equation: 

   ***Y = β1X + β0 + e***

    ▷ β1 is the coefficient of the independent variable (slope)

    ▷ β0 is the constant term or the y intercept.

    ▷ e is the error - the distance between actual value and model value

Then consider this data also as tuples of (1, 18), (2, 22), (3, 45), (4, 49), (5, 86)

    • We might want to fit a straight line to the given data

    • Assume to fit a line with the equation Y = β1X + β0

    • Our goal is to minimize errors

To minimize the amount of distance(errors), we need to find proper β1 and β0.We choose build a function :

   ***l(β1,β0) =1/n∑ i( f (xi) - yi)²***

which often referred to as a ***lost function***,.

        lost function has three common formula:
            (1)MSE(Mean Squared Error)
            (2)RMSE(Root Mean Squared Error)
            (3)Logloss(Cross Entorpy loss) 
        
In this case, we choose **Mean Squared Error**.So when we want to fit a line to given data, we need to minimize the lost function.

then, computing lost function:

        E(β1,β0) =1/n∑ i( f (xi) - yi)²
                 =1/n∑ i( β1*xi + β0 - yi)²
        l'(β1,β0)=1/n∑ i 2*( β1*xi² - xi*yi)
                 =2/n*(2m(1 + 4 + 9 + 16 + 25) - 2(18 + 44 + 135 + 196 + 430))
                 = 2/5*(110m - 1646)
Since cost function is a ”Convex” function, it means when its derivative is 0, the cost function hits bottom.
So loss minimized at m = 14.96.

>Regulation in lost function
>
>We will always face **over-fitting issue** in real problem. **over-fitting issue** is that the parameters of model are large and model's rebustness is poor, which means a little change of test data may cause a huge difference in result.So in order to aviod over-fitting,
>
>
>
>>first we need to remove parameters which have little contribution and generate sparse matrix, that is, the l1 norm( mean absolute error):

>>   ***l1 = l + α∑ i|βi|***
   
    where l is lost function, ∑ i|βi| is l1 regularizers, α is regularization coefficient, βi is parameters.
>>we can visualize l1 lost function：

>>![l1](https://i.loli.net/2018/11/28/5bfe89e366bba.jpg)

>>The contour line in the figure is that of l, and the black square is the graph of L1 function. The place where the contour line of l intersects the graph of L1 for the first time is the optimal solution. It is easy to find that the black square must intersect the contour line at the vertex of the square first. l is much more likely to contact those angles than it is to contact any other part. Some dimensions of these points are 0 which will make some features equal to 0 and generate a sparse matrix, which can then be used for feature selection.

>>Secondly, we can make parameters as little as possible by implement l2 norm:

   >>***l2 = l + α(∑ i|βi|²)^1/2*** 
    
    where l is lost function, (∑ i|βi|²)^1/2 is l2 regularizers, α is regularization coefficient, βi is parameters.
>>we can visualize l2 lost function：

>>![l2](https://i.loli.net/2018/11/28/5bfe89e366bba.jpg)

>>In comparison with the iterative formula without adding L2 regularization, parameters are multiplied by a factor less than 1 in each iteration, which makes parameters decrease continuously. Therefore, in general, parameters decreasing continuously.

### Optimization

Now we have a polynomial linear regression:

   ***y =β0 + β1x1 + β2x2 + ... + βdxd***

Similarly, we get the lost function :

   ***l(β0,β1...) =1/n∑ i( f (xi) - yi)²***
   
So in order to minimize the cost function, we need to choose each βi to minimize l(β0,β1...),this is what we called ***Gradient Descent***.

Gradient Descent is an iterative algorithm,Start from an initial guess and try to incrementally improve current solution,and at iteration step q(iter) is the current guess for βi.


#### How to calculate gradient

Suppose ▽l(β) is a vector whose ith entry is ith partial derivative evaluated at βi

![derivative](https://github.com/gnayoaixgnaw/machine_learning_project/blob/main/image/derivative1.png)

**Gradient Descent pseudocode**

![pseudocode](https://github.com/gnayoaixgnaw/machine_learning_project/blob/main/image/pseudocode1.png)

    • Here l is the ”learning rate” and controls speed of convergence
    • ▽l(β iter) is the gradient of L evaluated at iteration ”iter” with parameter of qiter
    • Stop conditions can be different
    
   **When to stop**

      Stop condition can be different, for example:
        • Maximum number of iteration is reached (iter < MaxIteration)
        • Gradient ▽l(β iter ) or parameters are not changing (||β(iter+1) - β(iter)|| < precisionValue)
        • Cost is not decreasing (||l(β(iter+1)) - L(β(iter))|| < precisionValue)
        • Combination of the above
        
**Gradient Descent calculation**

In privious sessions, we got the MSE, which is   
  

   ***l(β0,β1...) =1/n∑ i( f (xi) - yi)²***
   
then do expansion:  
  

   ***l(β0,β1...) =1/n∑ i( yi - (β0 + β1*x1+ β2*x2...+βd*xd))²***
  
   
Since it has mutiple dimensions,we compute partial derivatives:

![derivative](https://github.com/gnayoaixgnaw/machine_learning_project/blob/main/image/derivative.png)  


Now we can compute components of the gradients and then sum them up and update weights in the next iteration.
 
Here are more detailed pseudocode to compute gradient:

    // initialize parameters
    iteration = 0
    learning Rate = 0.01
    numIteration = X
    theta = np.random.normal(0, 0.1, d)

    while iteration < maxNumIteration:

      calculate gradients
      //update parameters
      theta -= learning Rate*gradients
      iteration+=1


**Learning rate**

The learning rate is a vital parameter in gradient descent as learning rate is responsible for convergence, if lr is small, convergent speed will be slow, on the contrary,when lr is large, function will converge very fast.

small learning rate:

![small_lr](https://github.com/gnayoaixgnaw/machine_learning_project/blob/main/image/lr_small.png) 

large learning rate:


![large_lr](https://github.com/gnayoaixgnaw/machine_learning_project/blob/main/image/lr_large.png) 

So how to find a proper learning rate? If we set lr a large value, the function will converge very fast at beginning but may miss the optimal solution,but if we set a small value, it will cost too much time to converge.

>Here I choose a very simple method, which name is ***Bold Driver*** to change learning rate dynamicly:
>
>At each iteration, compute the cost l(β0,β1...)
>
>Better than last time? 
>
>If cost decreases, increase learning rate
>l = 1.05 * l
>
>Worse than last time?
>
>l = 0.5 * l
>If cost increases, decrease rate

This graph illustrate the advantages of adaptive lr vs constant lr:

![different](https://github.com/gnayoaixgnaw/machine_learning_project/blob/main/image/adaptivelr.png) 


#### Optimizer

Variations of Gradient Descent depends on size of data that be used in each iteration:

      • Full Batch Gradient Descent (Using the whole data set (size n))
      • Stochastic Gradient Descent (SGD) (Using one sample per iteration (size 1))
      • Mini Batch Gradient Descent (Using a mini batch of data (size m < n))

**BGD** has a disadvantage: In each iteration, as it calculates gradients from whole dataset, this process will take lots of time.BGD can't overcome local minimum problem, because we can not add new data to trainning dataset.In other word, when function comes to local minimum point, full batch gradient will be 0, which means optimization process will stop.

**SGD** is always used in online situation for its speed.But since SGD uses only one sample in each iteration, the gradient can be affacted by noisy point, causeing a fact that function may not converge to optimal solution.

**MSGD** finds a trade-off between SGD and BGD.Different from BGD and SGD, MSGD only pick a small batch-size of data in each iteration, which not only minimized the impact from noisy point, but also reduced trainning time and increased the accuracy.

As iterations going on, we hope that learning rate becomes smaller.Because when function close to optimal solution, the changing step should be small to find best solution.So we need to gradually change learning rate.Also, we know that the weights for each coefficent is different, which means gradients of some coefficents are large while some are little.So in traditional SGD, changes of coefficents are not synchronous.So we need to distribute the coefficents which has large changes a small leanrning rate.

To deal with this issue, we can use **Adagrad(adaptive gradient algorithm)**

Adagrad will give each coefficent a proper learning rate:

![adagrad](https://github.com/gnayoaixgnaw/machine_learning_project/blob/main/image/adagrad1.png) 

    w : coefficents
    
    ∂l/∂w : gradient
    
    η: learning rate 
    
    h: sum of squares of all the previous gradients

when updating coefficent, we can adjust the scale by mutiplying 1/√h.

But as iteration going on, h will be very large, making updating step becomes very small. 

**RMSProp** can optimize this problem.RMSProp uses an exponential weighted average to eliminate swings in gradient descent: a larger derivative of a dimension means a larger exponential weighted average, and a smaller derivative means a smaller exponential weighted average. This ensures that the derivatives of each dimension are of the same order of magnitude, thus reducing swings:

    calculate gradient: 
      dwi = ∂L(w)/∂wi
    update h (add weight β, dropped parts of hwi)
      hwi =β * hwi + (1-β)*(dwi)²
    update wi (√hwi can be 0 some times, so we add a small value c to √hwi)
      wi = wi - η/(√hwi +c) * dwi 
***RMSProp code here***

···
def RMSprop(x, y, step=0.01, iter_count=500, batch_size=4, alpha=0.9, beta=0.9):
    length, features = x.shape
    data = np.column_stack((x, np.ones((length, 1))))
    w = np.zeros((features + 1, 1))
    Sdw, v, eta = 0, 0, 10e-7
    start, end = 0, batch_size
    
    # 开始迭代
    for i in range(iter_count):
        # 计算临时更新参数
        w_temp = w - step * v
        
        # 计算梯度
        dw = np.sum((np.dot(data[start:end], w_temp) - y[start:end]) * data[start:end], axis=0).reshape((features + 1, 1)) / length        
        
        # 计算累积梯度平方
        Sdw = beta * Sdw + (1 - beta) * np.dot(dw.T, dw)
        
        # 计算速度更新量、
        v = alpha * v + (1 - alpha) * dw
        
        # 更新参数
        w = w - (step / np.sqrt(eta + Sdw)) * v
        start = (start + batch_size) % length
        if start > length:
            start -= length
        end = (end + batch_size) % length
        if end > length:
            end -= length
    return w
···
	
 

