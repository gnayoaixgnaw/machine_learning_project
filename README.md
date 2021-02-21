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

Let's look at a simple linear regression equation: 

***Y = β1X + β0 + e***

    ▷ β1 is the coefficient of the independent variable (slope)

    ▷ β0 is the constant term or the y intercept.

    ▷ e is the error - the distance between actual value and model value

Then consider this data also as tuples of (1, 18), (2, 22), (3, 45), (4, 49), (5, 86)

    • We might want to fit a straight line to the given data

    • Assume to fit a line with the equation Y = β1X + β0

    • Our goal is to minimize errors

To minimize the amount of distance(errors), we need to find proper β1 and β0.We choose **least-squares fit**.
Least-squares fit gives a function:

***l(β1,β0) =∑ i( f (xi) - yi)²***

which often referred to as a ***lost function*** .So when we want to fit a line to given data, we need to minimize the lost function.

In this case, computing lost function:

        l(β1,β0) =∑ i( f (xi) - yi)²
                 =∑ i( β1*xi + β0 - yi)²
        l'(β1,β0)=∑ i 2*( β1*xi² - xi*yi)
                 =2m(1 + 4 + 9 + 16 + 25) 􀀀 2(18 + 44 + 135 + 196 + 430)
                 = 110m - 1646
Since cost function is a ”Convex” function, it means when its derivative is 0, the cost function hits bottom.
So loss minimized at m = 14.96.
