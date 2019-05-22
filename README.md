# Linear-Regression
Implement linear regression

### Introduction
Learning a linear regression model means estimating the values of the coefficients used in the representation with the data that we have available.

There are many techniques to build linear regression model. Ordinary Least Squares is the most common method used in general. Also take note of Gradient Descent as it is the most common technique taught in machine learning classes.

#### 1. Ordinary Least Squares
The Ordinary Least Squares procedure seeks to minimize the sum of the squared residuals. This means that given a regression line through the data we calculate the distance from each data point to the regression line, square it, and sum all of the squared errors together. This is the quantity that ordinary least squares seeks to minimize.

This approach treats the data as a matrix and uses linear algebra operations to estimate the optimal values for the coefficients. It means that all of the data must be available and you must have enough memory to fit the data and perform matrix operations. This procedure is very fast to calculate.

#### 2. Gradient Descent
Gradient Descent works by starting with random values for each coefficient. The sum of the squared errors are calculated for each pair of input and output values. A learning rate is used as a scale factor and the coefficients are updated in the direction towards minimizing the error. The process is repeated until a minimum sum squared error is achieved or no further improvement is possible.

When using this method, you must select a learning rate (alpha) parameter that determines the size of the improvement step to take on each iteration of the procedure.

### Requirement
* [numpy](https://www.numpy.org/)
* [pandas](https://pandas.pydata.org/)
* [scikit-learn](https://scikit-learn.org/stable/index.html)
* [matplotlib](https://matplotlib.org/)

### Regression Line Result
![Regression Line](https://github.com/shin7/Linear-Regression/blob/master/regression_line.png)
