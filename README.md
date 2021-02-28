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

   ![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign%7D%20l%28%5CTheta%20_0%2C%5CTheta%20_1%29%20%26%20%3D%20%5Cfrac%7B1%7D%7B2n%7D%20%5Csum_%7Bi%7D%20%28f%28x_%7Bi%7D%29-y_%7Bi%7D%29%5E%7B2%7D%20%5Cnonumber%20%5C%5C%20%26%3D%20%5Cfrac%7B1%7D%7B2n%7D%20%5Csum_%7Bi%7D%28%5CTheta%20_1%20x_i%20&plus;%20%5CTheta%20_1%20-%20y_%7Bi%7D%29%5E%7B2%7D%5Cnonumber%20%5Cend%7Balign%7D)
   
   
   
   
   ![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign%7D%20l%7B%7D%27%28%5CTheta%20_0%2C%5CTheta%20_1%29%20%26%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20%5Csum_%7Bi%7D%20%28%5CTheta%20_1%20x_%7Bi%7D%20%5E%7B2%7D%20-x_%7Bi%7Dy_%7Bi%7D%29%20%5Cnonumber%20%5C%5C%20%26%3D%20%5Cfrac%7B1%7D%7Bn%7D%20%5Csum_%7Bi%7D%282%5CTheta%20_1%281%20&plus;%204%20&plus;%209%20&plus;%2016%20&plus;%2025%29%20-%202%2818%20&plus;%2044%20&plus;%20135%20&plus;%20196%20&plus;%20430%29%29%5Cnonumber%20%5C%5C%20%26%3D%20%5Cfrac%7B1%7D%7B5%7D%28110m%20-%201646%29%5Cnonumber%20%5Cend%7Balign%7D)

Since cost function is a ”Convex” function, when its derivative is 0, the cost function hits bottom.
So loss minimized at θ = 14.96.

Now we have a polynomial linear regression, suppose each entity x has d dimensions:

![equation](https://latex.codecogs.com/gif.latex?y%20%3D%20%5CTheta_%7B0%7D%20&plus;%20%5CTheta_%7B1%7Dx_1%20&plus;%20...%20&plus;%20%5CTheta_%7Bd%7Dx_d)

Similarly, we get the lost function :

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign%7D%20l%28%5CTheta%20_0%2C%5CTheta%20_1...%5CTheta%20_d%29%20%26%20%3D%20%5Cfrac%7B1%7D%7B2n%7D%20%5Csum_%7Bi%7D%20%28f%28x_%7Bi%7D%29%20-%20y_%7Bi%7D%29%5E2%20%5Cnonumber%20%5Cend%7Balign%7D)
   
So in order to minimize the cost function, we need to choose each θi to minimize l(θ0,θ1...),this is what we called ***Gradient Descent***.

Gradient Descent is an iterative algorithm,Start from an initial guess and try to incrementally improve current solution,and at iteration step θ(iter) is the current guess for θi.


#### How to calculate gradient

Suppose ▽l(θ) is a vector whose ith entry is ith partial derivative evaluated at θi

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign%7D%20%5CDelta%20l%28%5Ctheta%29%20%3D%20%5Cbegin%7Bbmatrix%7D%5Cnonumber%20%5Cfrac%7B%5Cpartial%20l%28%5Ctheta%29%7D%7B%5Cpartial%20%5Ctheta%20_0%7D%5C%5C%20%5Cfrac%7B%5Cpartial%20l%28%5Ctheta%29%7D%7B%5Cpartial%20%5Ctheta%20_1%7D%5C%5C%20.%5C%5C%20.%5C%5C%20%5Cfrac%7B%5Cpartial%20l%28%5Ctheta%29%7D%7B%5Cpartial%20%5Ctheta%20_d%7D%5C%5C%20%5Cend%7Bbmatrix%7D%20%5Cend%7Balign%7D)

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
  

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign%7D%20l%28%5CTheta%20_0%2C%5CTheta%20_1...%5CTheta%20_d%29%20%26%20%3D%20%5Cfrac%7B1%7D%7B2n%7D%20%5Csum_%7Bi%7D%20%28f%28x_%7Bi%7D%29%20-%20y_%7Bi%7D%29%5E2%20%5Cnonumber%20%5Cend%7Balign%7D)
   
then do expansion:  
  

![equation](https://latex.codecogs.com/gif.latex?l%28%5Ctheta%20_0%2C%5Ctheta%20_1%2C...%2C%5Ctheta%20_d%29%3D%20%5Cfrac%7B1%7D%7B2n%7D%5Csum_%7Bi%7D%5E%7B%7D%28y%5E%7B%28i%29%7D-%20%28%5Ctheta%20_0&plus;%20%5Ctheta%20_1x_1%5E%7B%28i%29%7D&plus;...%5Ctheta%20_dx_d%5E%7B%28i%29%7D%29%29)  
    
Since it has mutiple dimensions,we compute partial derivatives:

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign%7D%20%5Cfrac%7B%5Cpartial%20l%28%5Ctheta%29%7D%7B%5Cpartial%20%5Ctheta%20_1%7D%20%3D%20-%20%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7B1%7D%5E%7Bn%7D%20x_%7B1%7D%5E%7B%28i%29%7D%28y%5E%7B%28i%29%7D%20-%20%28%5Ctheta%20_0&plus;%5Ctheta%20_1x_1%5E%7B%28i%29%7D%20&plus;%20...&plus;%20%5Ctheta%20_dx_d%5E%7B%28i%29%7D%20%29%5Cnonumber%5C%5C%20%5Cfrac%7B%5Cpartial%20l%28%5Ctheta%29%7D%7B%5Cpartial%20%5Ctheta%20_2%7D%20%3D%20-%20%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7B1%7D%5E%7Bn%7D%20x_%7B2%7D%5E%7B%28i%29%7D%28y%5E%7B%28i%29%7D%20-%20%28%5Ctheta%20_0&plus;%5Ctheta%20_1x_1%5E%7B%28i%29%7D%20&plus;%20...&plus;%20%5Ctheta%20_dx_d%5E%7B%28i%29%7D%20%29%5Cnonumber%5C%5C%20...%5Cnonumber%5C%5C%20%5Cfrac%7B%5Cpartial%20l%28%5Ctheta%29%7D%7B%5Cpartial%20%5Ctheta%20_d%7D%20%3D%20-%20%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7B1%7D%5E%7Bn%7D%20x_%7Bd%7D%5E%7B%28i%29%7D%28y%5E%7B%28i%29%7D%20-%20%28%5Ctheta%20_0&plus;%5Ctheta%20_1x_1%5E%7B%28i%29%7D%20&plus;%20...&plus;%20%5Ctheta%20_dx_d%5E%7B%28i%29%7D%20%29%5Cnonumber%20%5Cend%7Balign%7D)

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



## Regulation in lost function

We will always face **over-fitting issue** in real problem. **over-fitting issue** is that the parameters of model are large and model's rebustness is poor, which means a little change of test data may cause a huge difference in result.So in order to aviod over-fitting,

### l1 norm

We need to remove parameters which have little contribution and generate sparse matrix, that is, the l1 norm( mean absolute error):

![equation](https://latex.codecogs.com/gif.latex?l_1%20%3D%20l&plus;%5Clambda%20%5Csum_%7Bi%3D1%7D%5E%7Bd%7D%5Cleft%20%7C%20%5Ctheta%20_i%20%5Cright%20%7C)
   
    	where l is lost function, ∑ i|θi| is l1 regularizers, λ is regularization coefficient, θi is parameters.
we can visualize l1 lost function：

![l1](https://i.loli.net/2018/11/28/5bfe89e366bba.jpg)

The contour line in the figure is that of l, and the black square is the graph of L1 function. The place where the contour line of l intersects the graph of L1 for the first time is the optimal solution. It is easy to find that the black square must intersect the contour line at the vertex of the square first. l is much more likely to contact those angles than it is to contact any other part. Some dimensions of these points are 0 which will make some features equal to 0 and generate a sparse matrix, which can then be used for feature selection.

### l2 norm

We can make parameters as little as possible by implement l2 norm:

![equation](https://latex.codecogs.com/gif.latex?l_1%20%3D%20l&plus;%5Clambda%20%5Csum_%7Bi%3D1%7D%5E%7Bd%7D%5Cleft%20%7C%20%5Ctheta%20_i%20%5Cright%20%7C%5E%7B2%7D)


    	where l is lost function, ∑ i|θi|² is l2 regularizers, λ is regularization coefficient, θi is parameters.
we can visualize l2 lost function：

![l2](https://i.loli.net/2018/11/28/5bfe89e366bba.jpg)

In comparison with the iterative formula without adding L2 regularization, parameters are multiplied by a factor less than 1 in each iteration, which makes parameters decrease continuously. Therefore, in general, parameters decreasing continuously.

	
	
	
## Logistic regression

Logistic regression is supervised model especially for prediction problem.Suppose we have a prediction problem.It is natural to assume that output y (0/1) given the independent variable(s) X ,which has d dimensions and model parameter θ is sampled from the exponential family.

It makes sense to assume that the x is sampled from a Bernoulli and here is the log-likelihood:

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign%7D%20L%28p%7Cx_1%2Cx_2...%2Cx_n%29%20%26%3D%20%5Cprod_%7Bi%20%3D%201%7D%5E%7Bn%7Dp%5E%7Bx_i%7D%281-p%29%5E%7B%281-x_i%29%7D%5Cnonumber%20%5C%5C%20%26%3D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5Bx_i%5Clog%20%28p%29&plus;%281-x_i%29%5Clog%20%281-p%29%5D%20%5Cnonumber%20%5Cend%7Balign%7D)

Given a bunch of data for example,suppose output Y has (0/1):

	(92, 12), (23, 67), (67, 92), (98, 78), (18, 45), (6, 100)

	Final Result in class: 0, 0, 1, 1, 0, 0

	• If coefs are (-1, -1), LLH is -698
	• If coefs are (1, -1), LLH is -133
	• If coefs are (-1, 1), LLH is 7.4
	• If coefs are (1, 1), LLH is 394
	

However this is not enough to get the loss function, logistic regreesion needs a sigmoid function to show the probability of y = 0/1,which is :

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign%7D%20P%28x_i%29%20%26%3D%20%5Cfrac%7B1%7D%7B1-e%5E%7B-y_i%7D%7D%5Cnonumber%20%5C%5C%20%26%3D%5Cfrac%7Be%5E%7B%5Ctheta%20_0&plus;%5Ctheta_1%20x_1&plus;...&plus;%5Ctheta_d%20x_d%7D%7D%7B1&plus;e%5E%7B%5Ctheta%20_0&plus;%5Ctheta_1%20x_1&plus;...&plus;%5Ctheta_d%20x_d%7D%7D%20%5Cnonumber%20%5Cend%7Balign%7D)

The parameter ω is related to X that is, assuming X is vector-valued and ω can be represent as :

![equation](https://latex.codecogs.com/gif.latex?%5Comega%20_i%20%3D%20%5Csum_%7Bj%20%3D%201%7D%5E%7Bd%7Dx_j%5E%7B%28i%29%7D%20%5Ctheta%20_j)

     where θ is regression coefficent and j is entity's jth dimension .

Now its time to implement Log-likelihood in logistic regression, written as:

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign%7D%20L%28p%7Cx_1%2Cx_2...%2Cx_n%2C%20y_1%2Cy_2...%2Cy_n%29%20%26%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5By_i%5Clog%20%5Cfrac%7Be%5E%7B%5Comega%20_i%7D%7D%7B1&plus;e%5E%7B%5Comega_i%7D%7D&plus;%281-y_i%29%5Clog%20%281-%5Cfrac%7Be%5E%7B%5Comega_i%7D%7D%7B1&plus;e%5E%7B%5Comega_i%7D%7D%29%5D%5Cnonumber%20%5C%5C%20%26%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5By_i%28%5Clog%20e%5E%7B%5Comega_i%7D%29-%20%5Clog%20%281&plus;e%5E%7B%5Comega_i%7D%29%5D%5Cnonumber%20%5C%5C%20%26%3D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5By_i%5Comega_i-%5Clog%20%281&plus;e%5E%7B%5Comega_i%7D%29%5D%5Cnonumber%20%5Cend%7Balign%7D)

	
Now calculate loss function.As gradient descent need to minimize loss function,the loss function should be negative LLH:

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign%7D%20loss%20function%20%26%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5By_i%5Comega_i-%5Clog%20%281&plus;e%5E%7B%5Comega_i%7D%29%5D%5Cnonumber%20%5C%5C%20%26%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5B-%20y_i%5Comega_i%20&plus;%20%5Clog%20%281&plus;e%5E%7B%5Comega_i%7D%29%5D%5Cnonumber%20%5Cend%7Balign%7D)

Appling regularization (l2 norm):

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign%7D%20loss%20function%20%26%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5B-%20y_i%5Comega_i%20&plus;%20%5Clog%20%281&plus;e%5E%7B%5Comega_i%7D%29%5D%5Cnonumber%20%5C%5C%20%26%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5B-%20y_i%5Comega_i%20&plus;%20%5Clog%20%281&plus;e%5E%7B%5Comega_i%7D%29%5D%20&plus;%20%5Clambda%20%5Csum_%7Bi%3D1%7D%5E%7Bj%7D%5Ctheta%20_i%20%5E%7B2%7D%5Cnonumber%20%5Cend%7Balign%7D)

	where j is entity's jth dimension.


#### How to calculate gradient

Suppose θj is jth partial derivative :

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign%7D%20%5CDelta%20l%28%5Ctheta%29%20%3D%20%5Cbegin%7Bbmatrix%7D%5Cnonumber%20%5Cfrac%7B%5Cpartial%20l%28%5Ctheta%29%7D%7B%5Cpartial%20%5Ctheta%20_0%7D%5C%5C%20%5Cfrac%7B%5Cpartial%20l%28%5Ctheta%29%7D%7B%5Cpartial%20%5Ctheta%20_1%7D%5C%5C%20.%5C%5C%20.%5C%5C%20%5Cfrac%7B%5Cpartial%20l%28%5Ctheta%29%7D%7B%5Cpartial%20%5Ctheta%20_d%7D%5C%5C%20%5Cend%7Bbmatrix%7D%20%5Cend%7Balign%7D)

Since it has mutiple dimensions,we compute partial derivatives:

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign%7D%20%5Cfrac%7B%5Cpartial%20l%7D%7B%5Cpartial%20%5Ctheta%20_1%7D%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5By_ix_1%5E%7B%28i%29%7D&plus;x_1%5E%7B%28i%29%7D%5Cfrac%7Be%5E%7B%5Comega_i%7D%7D%7B1&plus;e%5E%7B%5Comega_i%7D%7D%5D&plus;2%5Clambda%20%5Comega%20_1%20%5Cnonumber%20%5C%5C%20%5Cfrac%7B%5Cpartial%20l%7D%7B%5Cpartial%20%5Ctheta%20_2%7D%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5By_ix_2%5E%7B%28i%29%7D&plus;x_2%5E%7B%28i%29%7D%5Cfrac%7Be%5E%7B%5Comega_i%7D%7D%7B1&plus;e%5E%7B%5Comega_i%7D%7D%5D&plus;2%5Clambda%20%5Comega%20_2%20%5Cnonumber%20%5C%5C%20...%20%5Cnonumber%20%5C%5C%20%5Cfrac%7B%5Cpartial%20l%7D%7B%5Cpartial%20%5Ctheta%20_d%7D%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5By_ix_d%5E%7B%28i%29%7D&plus;x_d%5E%7B%28i%29%7D%5Cfrac%7Be%5E%7B%5Comega_i%7D%7D%7B1&plus;e%5E%7B%5Comega_i%7D%7D%5D&plus;2%5Clambda%20%5Comega%20_d%20%5Cnonumber%20%5Cend%7Balign%7D)
	

**Gradient Descent pseudocode in Pyspark**

![pseudocode1](https://github.com/gnayoaixgnaw/machine_learning_project/blob/main/image/pseudocode2.png)


#### Implement code via Pyspark 

***Check [here](https://github.com/gnayoaixgnaw/Big_Data_Analytics/tree/main/assignment4)***


## Support victor machine

Traditional svm is binary-class svm, and there is also multi-class svm. 


### binary class svm 

Suppose there is a dataset that is linearly separable, it is possible to put a strip between two classes.The points that keep strip from expending are 'support vector'.

So basiclly, all points x in any line or plane or hyperplane can be discribed as a vevtor with distance b:

![equation](https://latex.codecogs.com/gif.latex?%5Cvec%7Bw%7D%5Ccdot%20x_0%20&plus;%20b%20%3D%200)

Now here is a point x and we need to caluculate the distance (which can be discribed as y) between this point and plane:

![equation](https://latex.codecogs.com/gif.latex?y%20%3D%20%5Cvec%7Bw%7D%5Ccdot%28x-x_0%29%20%3D%20%5Cvec%7Bw%7D%5Ccdot%20x%20-%20%5Cvec%7Bw%7D%5Ccdot%20x_0%20%3D%20%5Cvec%7Bw%7D%5Ccdot%20x&plus;b)

Notice y should be -1 or 1 to determine which of the sides the point x is.

This is because in basic svm, it choose two planes:

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign%7D%20%5Cvec%7Bw%7D%5Ccdot%20x&plus;b%20%3D%201%20%5Cnonumber%20%5C%5C%20%5Cvec%7Bw%7D%5Ccdot%20x&plus;b%20%3D%20-1%5Cnonumber%20%5Cend%7Balign%7D)

	where y > 1 then x is '+'(or x is positive sample) 
	where y < -1 then x is '-'(or x is negative sample)

![image](https://pic4.zhimg.com/v2-197913c461c1953c30b804b4a7eddfcc_1440w.jpg?source=172ae18b)

The distance between two planes are ![equation](https://latex.codecogs.com/gif.latex?%5Cfrac%7B2%7D%7B%5Cleft%20%5C%7C%20%5Cvec%7Bw%7D%20%5Cright%20%5C%7C%7D),so we need to maximize this distance to get optimal solution.

#### loss function

First, define a normal loss function:

![equation](https://latex.codecogs.com/gif.latex?l%20%3D%20%5Csum_%7Bi%20%3D%201%7D%5E%7Bn%7Dl%28y_i%20%5E%7Bpred%7D%2C%20y_i%5E%7Btrue%7D%29)

when this loss function is hinge loss, it is exactly svm's loss function:

![equation](https://latex.codecogs.com/gif.latex?l%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7Dmax%280%2C%20y_i%5E%7Bpred%7D*y_i%5E%7Btrue%7D%29%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7Dmax%280%2C%201%20-%5Cvec%7Bw%7D%5Ccdot%20x_i*y_i%29)

then add l2 norm, the final loss function is :

![equation](https://latex.codecogs.com/gif.latex?l%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7Dmax%280%2C%201%20-%5Cvec%7Bw%7D%5Ccdot%20x_i*y_i%29%20&plus;%20%5Clambda%20%5Cleft%20%5C%7C%20%5Cvec%7Bw%7D%20%5Cright%20%5C%7C%5E%7B2%7D)

#### How to calculate gradient

We can use Chain rule to compute the derivative:

![equation](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20l%7D%7B%5Cpartial%20%5Cvec%7Bw%7D%7D%20%3D%20%5Cfrac%7B%5Cpartial%20l%7D%7B%5Cpartial%20%28%5Cvec%7Bw%7D%5Ccdot%20x%29%7D%5Cfrac%7B%5Cpartial%20%28%5Cvec%7Bw%7D%5Ccdot%20x%29%7D%7B%5Cpartial%20%28%5Cvec%7Bw%7D%29%7D%20&plus;%202%5Clambda%20%5Cvec%7Bw%7D)

then calculate derivatives for two parts:

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign%7D%20%5Cfrac%7B%5Cpartial%20l%7D%7B%5Cpartial%20%28%5Cvec%7Bw%7D%5Ccdot%20x%29%7D%20%3D%20%5Cfrac%7B%5Cpartial%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7Dmax%280%2C%201-%20%5Cvec%7Bw%7D%5Ccdot%20x_i*y_i%20%5E%7Btrue%7D%29%7D%7B%5Cpartial%20%28%5Cvec%7Bw%7D%5Ccdot%20x_i%29%7D%20%5C%5C%20%5Cfrac%7B%5Cpartial%20%28%5Cvec%7Bw%7D%5Ccdot%20x%29%7D%7B%5Cpartial%20%5Cvec%7Bw%7D%7D%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7Dx_i%20%5Cend%7Balign%7D)

In conclusion, the final gradient is :
(1)if ![equation](https://latex.codecogs.com/gif.latex?%281-%20%5Cvec%7Bw%7D%5Ccdot%20x_i*y_i%20%5E%7Btrue%7D%29%20%3C%200):

below fomula(1) = 0, which means the derivative = 0

(2)if ![equation](https://latex.codecogs.com/gif.latex?%281-%20%5Cvec%7Bw%7D%5Ccdot%20x_i*y_i%20%5E%7Btrue%7D%29%20%3E%20%3D%200):

below fomula(1) = ![equation](https://latex.codecogs.com/gif.latex?-%20y_i%20%5E%7Btrue%7D), which means the derivative = ![equation](https://latex.codecogs.com/gif.latex?%5Csum_%7Bi%20%3D%200%7D%5E%7Bn%7D-%20y_i%20%5E%7Btrue%7Dx_i&plus;2%5Clambda%20%5Cvec%7Bw%7D)

then add l2 norm, the final derivatives can be written as:

![equation](https://latex.codecogs.com/gif.latex?%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%200&plus;%202%5Clambda%20%5Cvec%7Bw%7D%20%26%20%2Cif%20%281-%20y_i%20%5E%7Btrue%7D*%5Cvec%7Bw%7D%5Ccdot%20x_i%29%20%3C0%5C%5C%20%5Csum_%7Bi%20%3D%200%7D%5E%7Bn%7D-%20y_i%20%5E%7Btrue%7Dx_i&plus;2%5Clambda%20%5Cvec%7Bw%7D%26%20%2Cif%20%281-%20y_i%20%5E%7Btrue%7D*%5Cvec%7Bw%7D%5Ccdot%20x_i%29%20%3E%3D0%20%5Cend%7Bmatrix%7D%5Cright.)


#### Implement code via Pyspark 

***Check [here]()***

### multiple class svm 

Different from binary-class svm, multi-class svm's loss function requires the score on the correct class always be Δ(a boundary value) higher than scores on incorrect classes.

Suppose there is a dataset, the ith entity xi contains its features(has d features represent as vector) and class yi, then given svm model f(xi,w) to calculate the scores(as vector) in all classes,here we use si to represent this vector.

#### loss function

According to the definition of multi-class svm, we can get loss fucntion:

![equation](https://latex.codecogs.com/gif.latex?l_i%20%3D%20%5Csum_%7Bi%5Cneq%20y_i%7D%5E%7Bd%7Dmax%280%2Cs_j%20-%20s_y_i&plus;%5CDelta%20%29)

For example:

Suppose there are 3 classes, for the ith sample xi, we get scores = [12,-7,10],where the first class(yi) is correct.Then we make Δ=9, applying the below loss function, we can calculate the loss of xi:

![equation](https://latex.codecogs.com/gif.latex?l_i%20%3D%20max%280%2C%20-7-12&plus;9%29&plus;max%280%2C%2010-12&plus;9%29)

As w is a vector in loss function, we expend loss function:

![equation](https://latex.codecogs.com/gif.latex?l_i%20%3D%20%5Csum_%7Bi%5Cneq%20y_i%7D%5E%7Bd%7Dmax%280%2C%5Cvec%7Bw_j%7D%5E%7BT%7Dx_i%20-%20%5Cvec%7Bw_y_i%7D%5E%7BT%7Dx_i%20&plus;%5CDelta%20%29)

Then add l2 norm:

![equation](https://latex.codecogs.com/gif.latex?l_i%20%3D%20%5Csum_%7Bi%5Cneq%20y_i%7D%5E%7Bd%7Dmax%280%2C%5Cvec%7Bw_j%7D%5E%7BT%7Dx_i%20-%20%5Cvec%7Bw_y_i%7D%5E%7BT%7Dx_i%20&plus;%5CDelta%29%20&plus;%20%5Clambda%20%5Cleft%20%5C%7C%20%5Cvec%7Bw%7D%20%5Cright%20%5C%7C%5E2)

#### How to calculate gradient

As yi and j are different, we calculate wyi's derivative for ith sample xi first:

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign%7D%20%5Cfrac%7B%5Cpartial%20l_i%7D%7B%5Cpartial%20%5Cvec%7Bw_y_i%7D%7D%20%26%3D%20%5Cfrac%7B%5Cpartial%20%5Csum_%7Bi%5Cneq%20y_i%7D%5E%7Bd%7Dmax%280%2C%5Cvec%7Bw_j%7D%5E%7BT%7Dx_i%20-%20%5Cvec%7Bw_y_i%7D%5E%7BT%7Dx_i%20&plus;%5CDelta%29%20&plus;%20%5Clambda%20%5Cleft%20%5C%7C%20%5Cvec%7Bw%7D%20%5Cright%20%5C%7C%5E2%7D%7B%5Cpartial%20%5Cvec%7Bw_y_i%7D%7D%20%5Cnonumber%5C%5C%20%26%3D%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20-x_i&plus;%202%5Clambda%20%5Cvec%7Bw_y_i%7D%26%2C%20if%20%28%5Cvec%7Bw_j%7D%5E%7BT%7Dx_i%20-%20%5Cvec%7Bw_y_i%7D%5E%7BT%7Dx_i%20&plus;%5CDelta%29%20%3E%3D0%5C%5C%200&plus;2%5Clambda%20%5Cvec%7Bw_y_i%7D%26%20%2Cif%20%28%5Cvec%7Bw_j%7D%5E%7BT%7Dx_i%20-%20%5Cvec%7Bw_y_i%7D%5E%7BT%7Dx_i%20&plus;%5CDelta%29%20%3C0%20%5Cend%7Bmatrix%7D%5Cright.%20%5Cnonumber%20%5Cend%7Balign%7D)

then calculate wj's derivative for ith sample xi:

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign%7D%20%5Cfrac%7B%5Cpartial%20l_i%7D%7B%5Cpartial%20%5Cvec%7Bw_j%7D%7D%20%26%3D%20%5Cfrac%7B%5Cpartial%20%5Csum_%7Bi%5Cneq%20y_i%7D%5E%7Bd%7Dmax%280%2C%5Cvec%7Bw_j%7D%5E%7BT%7Dx_i%20-%20%5Cvec%7Bw_y_i%7D%5E%7BT%7Dx_i%20&plus;%5CDelta%29%20&plus;%20%5Clambda%20%5Cleft%20%5C%7C%20%5Cvec%7Bw%7D%20%5Cright%20%5C%7C%5E2%7D%7B%5Cpartial%20%5Cvec%7Bw_j%7D%7D%20%5Cnonumber%5C%5C%20%26%3D%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20x_i&plus;%202%5Clambda%20%5Cvec%7Bw_j%7D%26%2C%20if%20%28%5Cvec%7Bw_j%7D%5E%7BT%7Dx_i%20-%20%5Cvec%7Bw_y_i%7D%5E%7BT%7Dx_i%20&plus;%5CDelta%29%20%3E%3D0%5C%5C%200&plus;2%5Clambda%20%5Cvec%7Bw_j%7D%26%20%2Cif%20%28%5Cvec%7Bw_j%7D%5E%7BT%7Dx_i%20-%20%5Cvec%7Bw_y_i%7D%5E%7BT%7Dx_i%20&plus;%5CDelta%29%20%3C0%20%5Cend%7Bmatrix%7D%5Cright.%20%5Cnonumber%20%5Cend%7Balign%7D)

The gradient of vector w contians wj(from 1 to d) and each wj's value depends on the value of ![equation](https://latex.codecogs.com/gif.latex?%28%5Cvec%7Bw_j%7D%5E%7BT%7Dx_i%20-%20%5Cvec%7Bw_y_i%7D%5E%7BT%7Dx_i%20&plus;%5CDelta%29):

![equation](https://latex.codecogs.com/gif.latex?%5CDelta%20w_i%20%3D%20%5Bw_1%5E%7Bi%7D%3D%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20x_i%5C%5C%20-x_i%5C%5C%200%20%5Cend%7Bmatrix%7D%5Cright....w_1%5E%7Bi%20%7D%20%3D%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20x_i%5C%5C%20-x_i%5C%5C%200%20%5Cend%7Bmatrix%7D%5Cright.%5D)

So the gradient for batch of data can be written as:

![equation](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20l%7D%7B%5Cpartial%20%5Cvec%7Bw%7D%7D%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5CDelta%20%5Cvec%7Bw_i%7D%20&plus;%202%5Clambda%20%5Cvec%7Bw%7D)

#### Implement code via Pyspark 

***Check [here]()***




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

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign%7D%20h_j%20%5Cleftarrow%20h_j%20&plus;%28%5Cfrac%7B%5Cpartial%20l%7D%7B%5Cpartial%20%5Ctheta%20_j%7D%29%5E%7B2%7D%20%5C%5C%20%5Ctheta%20_j%20%5Cleftarrow%20%5Ctheta%20_j%20-%20%5Ceta%20%5Cfrac%7B1%7D%7B%5Csqrt%7Bh_j%7D%7D%5Cfrac%7B%5Cpartial%20l%7D%7B%5Cpartial%20%5Ctheta%20_j%7D%20%5Cend%7Balign%7D) 

    θ : coefficents
    
    ∂l/∂θ : gradient
    
    η: learning rate 
    
    hj: sum of squares of all the previous θj's gradients

when updating coefficent, we can adjust the scale by mutiplying 1/√h.

But as iteration going on, h will be very large, making updating step becomes very small. 

### RMSProp

**RMSProp** can optimize this problem.RMSProp uses an exponential weighted average to eliminate swings in gradient descent: a larger derivative of a dimension means a larger exponential weighted average, and a smaller derivative means a smaller exponential weighted average. This ensures that the derivatives of each dimension are of the same order of magnitude, thus reducing swings:

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign%7D%20h_j%20%5Cleftarrow%20%5Cbeta%20h_j%20&plus;%281-%5Cbeta%29%28%5Cfrac%7B%5Cpartial%20l%7D%7B%5Cpartial%20%5Ctheta%20_j%7D%29%5E%7B2%7D%20%5C%5C%20%5Ctheta%20_j%20%5Cleftarrow%20%5Ctheta%20_j%20-%20%5Ceta%20%5Cfrac%7B1%7D%7B%5Csqrt%7Bh_j&plus;c%7D%7D%5Cfrac%7B%5Cpartial%20l%7D%7B%5Cpartial%20%5Ctheta%20_j%7D%20%5Cend%7Balign%7D)

    √hj can be 0 some times, so we add a small value c to √hj
      
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

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign%7D%20h1_j%20%5Cleftarrow%20%5Cbeta1%20h_j%20&plus;%281-%5Cbeta1%29%28%5Cfrac%7B%5Cpartial%20l%7D%7B%5Cpartial%20%5Ctheta%20_j%7D%29%20%5Cnonumber%5C%5C%20h2_j%20%5Cleftarrow%20%5Cbeta2%20h_j%20&plus;%281-%5Cbeta2%29%28%5Cfrac%7B%5Cpartial%20l%7D%7B%5Cpartial%20%5Ctheta%20_j%7D%29%5E%7B2%7D%20%5Cnonumber%20%5Cend%7Balign%7D)
	
    If h1 and h2 are initialized to the 0 vectors, they are biased to 0, so bias correction is done to offset these biases by calculating the bias corrected h1 and h2:

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign%7D%20h1_j%20%5Cleftarrow%20%5Cfrac%7Bh1_j%7D%7B1%20-%5Cbeta1%20%5E%7Bt%7D%7D%20%5Cnonumber%5C%5C%20h2_j%20%5Cleftarrow%20%5Cfrac%7Bh2_j%7D%7B1%20-%5Cbeta2%20%5E%7Bt%7D%7D%20%5Cnonumber%5C%5C%20%5Ctheta%20_i%20%5Cleftarrow%20%5Ctheta%20_i%20-%20%5Cfrac%7B%5Cmu%20%7D%7B%5Csqrt%7Bh2_j%7D&plus;c%7Dh1_j%20%5Cnonumber%20%5Cend%7Balign%7D)
    
    t means t th iteration.
    
    
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



 

