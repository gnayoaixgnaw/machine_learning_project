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

Linear regression is very simple and basic.First, linear regression is supervised model, which means data should be labelled.Linear regression will find the relationships between features(x1,x2,x3....), which represent as coefficients of these variables.


### simple linear regression


Let's look at a simple linear regression equation: 

   ![equation](https://latex.codecogs.com/gif.latex?y%20%3D%20%5CTheta%20_1x&plus;%5CTheta%20_0)

    ▷ θ1 is the coefficient of the independent variable (slope)

    ▷ θ0 is the constant term or the y intercept.


Then consider this dataset as tuples of (1, 18), (2, 22), (3, 45), (4, 49), (5, 86)

    • We might want to fit a straight line to the given data

    • Assume to fit a line with the equation Y = θ1X + θ0

    • Our goal is to minimize errors

To minimize the amount of distance(errors), we need to find proper θ1 and θ0.We build a function ,which often referred to as a ***lost function***,.

	lost function has three common formula:
	    (1)MSE(Mean Squared Error)
	    (2)RMSE(Root Mean Squared Error)
	    (3)Logloss(Cross Entorpy loss) 
        

In this case, we choose **Mean Squared Error**.

![equation](https://latex.codecogs.com/gif.latex?l%28%5CTheta%20_0%2C%5CTheta%20_1%29%20%3D%20%5Cfrac%7B1%7D%7B2n%7D%5Csum_%7Bi%7D%20%28f%28x_%7Bi%7D%29-y_%7Bi%7D%29%5E%7B2%7D)

So when we want to fit a line to given data, we need to minimize the lost function.

then, computing lost function:

        E(θ1,θ0) =1/n∑ i( f (Xi) - Yi)²
                 =1/n∑ i( θ1*Xi + θ0 - Yi)²
        l'(θ1,θ0)=1/n∑ i 2*( θ1*Xi² - Xi*Yi)
                 =2/n*(2m(1 + 4 + 9 + 16 + 25) - 2(18 + 44 + 135 + 196 + 430))
                 = 2/5*(110m - 1646)
Since cost function is a ”Convex” function, when its derivative is 0, the cost function hits bottom.
So loss minimized at m = 14.96.

Now we have a polynomial linear regression:

   ***Y =θ0 + θ1X1 + θ2X2 + ... + θdXd***

Similarly, we get the lost function :

   ***l(θ0,θ1...) =1/n∑ i( f (X1,X2...Xd) - Yi)²***
   
So in order to minimize the cost function, we need to choose each θi to minimize l(θ0,θ1...),this is what we called ***Gradient Descent***.

Gradient Descent is an iterative algorithm,Start from an initial guess and try to incrementally improve current solution,and at iteration step θ(iter) is the current guess for θi.


#### How to calculate gradient

Suppose ▽l(θ) is a vector whose ith entry is ith partial derivative evaluated at θi

![derivative](https://github.com/gnayoaixgnaw/machine_learning_project/blob/main/image/derivative1.png)

**Gradient Descent pseudocode**

![pseudocode](https://github.com/gnayoaixgnaw/machine_learning_project/blob/main/image/pseudocode1.png)

    • Here l is the ”learning rate” and controls speed of convergence
    • ▽l(θ iter) is the gradient of L evaluated at iteration ”iter” with parameter of qiter
    • Stop conditions can be different
    
   **When to stop**

      Stop condition can be different, for example:
        • Maximum number of iteration is reached (iter < MaxIteration)
        • Gradient ▽l(θ iter ) or parameters are not changing (||θ(iter+1) - θ(iter)|| < precisionValue)
        • Cost is not decreasing (||l(θ(iter+1)) - L(θ(iter))|| < precisionValue)
        • Combination of the above
        
**Gradient Descent calculation**

In privious sessions, we got the loss function, which is   
  

   ***l(θ0,θ1...) =1/n∑ i( f (Xi) - Yi)²***
   
then do expansion:  
  

   ***l(θ0,θ1...) =1/n∑ i( Yi - (θ0 + θ1X1+ θ2X2...+θdXd))²***
  
   
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



#### Implement code via Pyspark 

***Check [here](https://github.com/gnayoaixgnaw/Big_Data_Analytics/tree/main/assignment3)***


## Logistic regression

Logistic regression is supervised model especially for prediction problem.Suppose we have a prediction problem.It is natural to assume that output y given the independent variable(s) X and model parameter θ is sampled from the exponential family.

The parameter θ is linearly related to X that is, assuming X is vector-valued

***θ = ∑j XjRj***

     where r is regression coefficent.

Then I implement Log-likelihood, written as:

***LLH(R1, R2, ..., Rd|x1, x2, ..., xn, y1, y2, ..., yn) = ∑i (logb(Yi) + θiT(Y) - f(θi))***


Given a bunch of data for example,suppose output Y has (0/1):

	(92, 12), (23, 67), (67, 92), (98, 78), (18, 45), (6, 100)

	Final Result in class: 0, 0, 1, 1, 0, 0

	• If coefs are (-1, -1), LLH is -698
	• If coefs are (1, -1), LLH is -133
	• If coefs are (-1, 1), LLH is 7.4
	• If coefs are (1, 1), LLH is 394
	
Now simplify this formula:

***LLH = ∑i (Yiθi - log(1+e^θi))***

     where θi = ∑j Rj*Xi,j ,i means ith entity,j means entity's ith dimension.

	
So now I can calculate loss function.As gradient descent need to minimize loss function,the loss function should be negative LLH:

***loss function = ∑i (-Yiθi + log(1+e^θi))***


Appling regularization (l2 norm):

***loss function = ∑i (-Yiθi + log(1+e^θi)) + λ∑j Rj^2***


#### How to calculate gradient

Suppose Rj is jth partial derivative evaluated at Rj:

![derivative2](https://github.com/gnayoaixgnaw/machine_learning_project/blob/main/image/derivative2.png)

	where Xi,j' is the jth dimension of Xi,Xi is ith entity in dataset.
	

**Gradient Descent pseudocode in Pyspark**

![pseudocode1](https://github.com/gnayoaixgnaw/machine_learning_project/blob/main/image/pseudocode2.png)


#### Implement code via Pyspark 

***Check [here](https://github.com/gnayoaixgnaw/Big_Data_Analytics/tree/main/assignment4)***


## Support victor machine
(working on it......)


## Regulation in lost function

We will always face **over-fitting issue** in real problem. **over-fitting issue** is that the parameters of model are large and model's rebustness is poor, which means a little change of test data may cause a huge difference in result.So in order to aviod over-fitting,

### l1 norm

We need to remove parameters which have little contribution and generate sparse matrix, that is, the l1 norm( mean absolute error):

***l1 = l + α∑ i|θi|***
   
    	where l is lost function, ∑ i|θi| is l1 regularizers, α is regularization coefficient, θi is parameters.
we can visualize l1 lost function：

![l1](https://i.loli.net/2018/11/28/5bfe89e366bba.jpg)

The contour line in the figure is that of l, and the black square is the graph of L1 function. The place where the contour line of l intersects the graph of L1 for the first time is the optimal solution. It is easy to find that the black square must intersect the contour line at the vertex of the square first. l is much more likely to contact those angles than it is to contact any other part. Some dimensions of these points are 0 which will make some features equal to 0 and generate a sparse matrix, which can then be used for feature selection.

### l2 norm

We can make parameters as little as possible by implement l2 norm:

   ***l2 = l + α(∑ i|θi|^2)^1/2*** 
    
    	where l is lost function, (∑ i|θi|²)^1/2 is l2 regularizers, α is regularization coefficient, θi is parameters.
we can visualize l2 lost function：

![l2](https://i.loli.net/2018/11/28/5bfe89e366bba.jpg)

In comparison with the iterative formula without adding L2 regularization, parameters are multiplied by a factor less than 1 in each iteration, which makes parameters decrease continuously. Therefore, in general, parameters decreasing continuously.

### ln norm 

ln norm is a general formula in norm:

   ***ln = l + α(∑ i|θi|^n)^1/n*** 
   
   	when n = 1 it is l1 norm, n =2 it is l2 norm
	


## Optimizer


Variations of Gradient Descent depends on size of data that be used in each iteration:

      • Full Batch Gradient Descent (Using the whole data set (size n))
      • Stochastic Gradient Descent (SGD) (Using one sample per iteration (size 1))
      • Mini Batch Gradient Descent (Using a mini batch of data (size m < n))

**BGD** has a disadvantage: In each iteration, as it calculates gradients from whole dataset, this process will take lots of time.BGD can't overcome local minimum problem, because we can not add new data to trainning dataset.In other word, when function comes to local minimum point, full batch gradient will be 0, which means optimization process will stop.

**SGD** is always used in online situation for its speed.But since SGD uses only one sample in each iteration, the gradient can be affacted by noisy point, causeing a fact that function may not converge to optimal solution.

**MSGD** finds a trade-off between SGD and BGD.Different from BGD and SGD, MSGD only pick a small batch-size of data in each iteration, which not only minimized the impact from noisy point, but also reduced trainning time and increased the accuracy.

### Learning rate 

The learning rate is a vital parameter in gradient descent as learning rate is responsible for convergence, if lr is small, convergent speed will be slow, on the contrary,when lr is large, function will converge very fast.

small learning rate:

![small_lr](https://github.com/gnayoaixgnaw/machine_learning_project/blob/main/image/lr_small.png) 

large learning rate:


![large_lr](https://github.com/gnayoaixgnaw/machine_learning_project/blob/main/image/lr_large.png) 

So how to find a proper learning rate? If we set lr a large value, the function will converge very fast at beginning but may miss the optimal solution,but if we set a small value, it will cost too much time to converge.

>Here I choose a very simple method, which name is ***Bold Driver*** to change learning rate dynamicly:
>
>At each iteration, compute the cost l(θ0,θ1...)
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



As iterations going on, we hope that learning rate becomes smaller.Because when function close to optimal solution, the changing step should be small to find best solution.So we need to gradually change learning rate.Also, we know that the weights for each coefficent is different, which means gradients of some coefficents are large while some are little.So in traditional SGD, changes of coefficents are not synchronous.So we need to distribute the coefficents which has large changes a small leanrning rate.

To deal with this issue, we can use **Adagrad(adaptive gradient algorithm)**, **RMSProp**, **Adam**

### Adagrad

Adagrad will give each coefficent a proper learning rate:

![adagrad](https://github.com/gnayoaixgnaw/machine_learning_project/blob/main/image/adagrad1.png) 

    w : coefficents
    
    ∂l/∂w : gradient
    
    η: learning rate 
    
    h: sum of squares of all the previous gradients

when updating coefficent, we can adjust the scale by mutiplying 1/√h.

But as iteration going on, h will be very large, making updating step becomes very small. 

### RMSProp

**RMSProp** can optimize this problem.RMSProp uses an exponential weighted average to eliminate swings in gradient descent: a larger derivative of a dimension means a larger exponential weighted average, and a smaller derivative means a smaller exponential weighted average. This ensures that the derivatives of each dimension are of the same order of magnitude, thus reducing swings:

    calculate gradient: 
      dwi = ∂L(w)/∂wi
    update h (add weight β, dropped parts of hwi)
      hwi =β * hwi + (1-β)*(dwi)²
    update wi (√hwi can be 0 some times, so we add a small value c to √hwi)
      wi = wi - η/(√hwi +c) * dwi 
***RMSProp code here***

```
def RMSprop(x, y, lr=0.01, iter_count=500, batch_size=4, beta=0.9):
    length, features = x.shape
    data = np.column_stack((x, np.ones((length, 1))))
    w = np.zeros((features + 1, 1))
    h, eta = 0, 10e-7
    start, end = 0, batch_size
    for i in range(iter_count):
        # calculate gradient
        dw = np.sum((np.dot(data[start:end], w) - y[start:end]) * data[start:end], axis=0) / length        
        # calculate sum of square of gradients
        h = beta * h + (1 - beta) * np.dot(dw, dw)                     
        # update w
        w = w - (lr / np.sqrt(eta + h)) * dw.reshape((features + 1, 1))
	
        start = (start + batch_size) % length
        if start > length:
            start -= length
        end = (end + batch_size) % length
        if end > length:
            end -= length
    return w
```
### Adam

***Adam*** is another powerful optimizer.It not only saved the sum of square of history gradients(h2 )but also save sum of history gradients(h1 ):

    calculate gradient: 
      dwi = ∂L(w)/∂wi
    update h1 and h2(add weight β1 and β2, dropped parts of hwi)
      h1wi =β1 * h1wi + (1-β1)*(dwi)
      h2wi =β2 * h2wi + (1-β2)*(dwi)²
    update wi (√hwi can be 0 some times, so we add a small value c to √hwi)
      wi = wi - η/(√hwi +c) * dwi 
	
    If h1 and h2 are initialized to the 0 vectors, they are biased to 0, so bias correction is done to offset these biases by calculating the bias corrected h1 and h2:

      h1wi' = h1/(1-β1wi)
      h2wi' = h2/(1-β2wi)
    
    then update wi:
      wi = wi - η/(√h2wi' +c) * h1wi'
***Adam code here***

```
def Adam(x, y, lr=0.01, iter_count=500, batch_size=4, beta1=0.9,beta2 = 0.999):
    length, features = x.shape
    data = np.column_stack((x, np.ones((length, 1))))
    w = np.zeros((features + 1, 1))
    h1, h2,eta = 0, 0,10e-7
    start, end = 0, batch_size
    for i in range(iter_count):
        # calculate gradient
        dw = np.sum((np.dot(data[start:end], w) - y[start:end]) * data[start:end], axis=0) / length        
        # calculate sums
        h1 = beta1 * h + (1 - beta1) * dw
	h2 = beta2 * h + (1 - beta2) * np.dot(dw, dw)
	# bias correction
	h1 = h1/(1- beta1)
	h2 = h2/(1- beta2)
        # update w
        w = w - (lr / np.sqrt(eta + h2)) * h1.reshape((features + 1, 1))
	
        start = (start + batch_size) % length
        if start > length:
            start -= length
        end = (end + batch_size) % length
        if end > length:
            end -= length
    return w
```

Now I tested these optimizers on MNIST and IMDB movie reviews,lets see the differenes of those optimizers:

for MNIST:

![mnist](https://github.com/gnayoaixgnaw/machine_learning_project/blob/main/image/mnist.png) 

for IMDB:

![imbd](https://github.com/gnayoaixgnaw/machine_learning_project/blob/main/image/imbd.png) 


### Now I will share a kaggle project based on ML:

***kaggel link :***

https://www.kaggle.com/uciml/mushroom-classification


This project is mushroom classification. There are about 2 categories (edible or poisonous) and 8124 records (52% edible and 48% poisonous).
For this project, I have tried three machine learning models:
SVM , Random forest and logistic regression. 
All three models have good performances:
Compare recall, precision, f1-score for both class
F1_score > 95%
Accuracy > 95%

This dataset includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus 
and Lepiota Family Mushroom drawn from The Audubon Society Field Guide to North American Mushrooms (1981). 
Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended. 
This latter class was combined with the poisonous one. The Guide clearly states that there is no simple rule for determining the edibility of a mushroom; 
no rule like "leaflets three, let it be'' for Poisonous Oak and Ivy.

***Code***

#### Check [here](https://github.com/gnayoaixgnaw/machine_learning_project/blob/main/mashroom/cs677project.ipynb)



 

