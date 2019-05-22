import numpy as np
import matplotlib.pyplot as plt


# Root Mean Squared Error
def rmse(x, y, beta):
    m, rmse = x.shape[0], 0
    for i in range(m):
        y_pred = beta[0] + beta[1] * x[i]
        rmse += (y[i] - y_pred) ** 2
    rmse = np.sqrt(rmse / m)
    print('RMSE = {}'.format(rmse))
    return rmse


# Coefficient of Determination(R^2 Score)
def r2_score(x, y, beta):
    m = x.shape[0]
    mean_y = np.mean(y)
    ss_t = 0  # ss_t is the total sum of squares
    ss_r = 0  # ss_r is the total sum of squares of residuals
    for i in range(m):
        y_pred = beta[0] + beta[1] * x[i]
        ss_t += (y[i] - mean_y) ** 2
        ss_r += (y[i] - y_pred) ** 2
    r2 = 1 - (ss_r / ss_t)
    print('R2_Score = {}'.format(r2))
    return r2


def predict(x, beta):
    y_pred = beta[0] + beta[1] * x
    print("Predicted value = {}".format(y_pred))
    return y_pred


# Plot the regression line
def plot_regression_line(X, y, beta=None, title='My Plot', xlabel='X', ylabel='Y'):
    # plotting the actual points as scatter plot
    plt.scatter(X, y, color='b', marker='o', s=20, label='Scatter Plot')

    # predicted response vector
    y_pred = beta[0] + beta[1] * X

    # plotting the regression line
    plt.plot(X, y_pred, color="k", label='Regression Line')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()
