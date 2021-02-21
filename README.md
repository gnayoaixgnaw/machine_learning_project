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

>simple linear regression
>dsasad
>saddd


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

   ***l(β1,β0) =∑ i( f (xi) - yi)²***

which often referred to as a ***lost function***,.

        lost function has three common formula:
            (1)MSE(Mean Squared Error)
            (2)RMSE(Root Mean Squared Error)
            (3)Logloss(Cross Entorpy loss) 
        
In this case, we choose **Mean Squared Error**.So when we want to fit a line to given data, we need to minimize the lost function.

In this case, computing lost function:

        E(β1,β0) =1/n∑ i( f (xi) - yi)²
                 =1/n∑ i( β1*xi + β0 - yi)²
        l'(β1,β0)=1/n∑ i 2*( β1*xi² - xi*yi)
                 =2/n*(2m(1 + 4 + 9 + 16 + 25) - 2(18 + 44 + 135 + 196 + 430))
                 = 2/5*(110m - 1646)
Since cost function is a ”Convex” function, it means when its derivative is 0, the cost function hits bottom.
So loss minimized at m = 14.96.

#### additional knowlage in lost function
We will always face **over-fitting issue** in real problem. **over-fitting issue** is that the parameters of model are large and model's rebustness is poor, which means a little change of test data may cause a huge difference in result.So in order to aviod over-fitting,

first we need to remove parameters which have little contribution and generate sparse matrix, that is, the l1 norm( mean absolute error):

   ***l1 = l + α∑ i|βi|***
   
    where l is lost function, ∑ i|βi| is l1 regularizers, α is regularization coefficient, βi is parameters.
we can visualize l1 lost function：

![l1](https://i.loli.net/2018/11/28/5bfe89e366bba.jpg)

The contour line in the figure is that of l, and the black square is the graph of L1 function. The place where the contour line of l intersects the graph of L1 for the first time is the optimal solution. It is easy to find that the black square must intersect the contour line at the vertex of the square first. l is much more likely to contact those angles than it is to contact any other part. Some dimensions of these points are 0 which will make some features equal to 0 and generate a sparse matrix, which can then be used for feature selection.

Secondly, we can make parameters as little as possible by implement l2 norm:

   ***l2 = l + α(∑ i|βi|²)^1/2*** 
    
    where l is lost function, (∑ i|βi|²)^1/2 is l2 regularizers, α is regularization coefficient, βi is parameters.
we can visualize l2 lost function：

![l2](https://i.loli.net/2018/11/28/5bfe89e366bba.jpg)

In comparison with the iterative formula without adding L2 regularization, parameters are multiplied by a factor less than 1 in each iteration, which makes parameters decrease continuously. Therefore, in general, parameters decreasing continuously.

### Optimization



