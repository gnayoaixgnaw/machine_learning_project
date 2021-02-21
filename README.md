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

To minimize the amount of distance(errors), we need to find proper β1 and β0. In this case, we choose **least-squares fit**.
Least-squares fit means, we build a function:

***l(β1,β0) =∑ i( f (xi) - xi)²***

This function often referred to as a ***lost function*** .So when we want to fit a line to given data, we need to minimize the lost function: min l(β1,β0).

