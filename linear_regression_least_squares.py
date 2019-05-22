# Implement Simple Linear Regression with 
# Ordinary Least Square Method Approach
# ========================================
import numpy as np
import pandas as pd
from utils import predict, rmse, r2_score, plot_regression_line


# Estimate the coefficients(beta0, beta1)
def estimate_coefficients(x, y):
    # number of sample
    m = x.shape[0]

    # mean of x and y vector
    mean_x, mean_y = np.mean(x), np.mean(y)

    # calculating cross-deviation and deviation about x
    sum_cross_deviation_xy, sum_squared_deviations_x = 0, 0

    for i in range(m):
        sum_cross_deviation_xy += (x[i] - mean_x) * (y[i] - mean_y)
        sum_squared_deviations_x += (x[i] - mean_x) ** 2

    # calculating regression coefficients
    beta1 = sum_cross_deviation_xy / sum_squared_deviations_x
    beta0 = mean_y - beta1 * mean_x
    print('Coefficients: beta0 = {}, beta1 = {}'.format(beta0, beta1))
    return beta0, beta1


def main():
    # read data from csv file
    data = pd.read_csv('headbrain.csv')
    print("data.shape = {}".format(data.shape))

    # load data to x and y
    x = data['Head Size(cm^3)'].values
    y = data['Brain Weight(grams)'].values
    print(x.shape)

    beta = estimate_coefficients(x, y)

    # TEST
    predict(3000, beta)
    # END TEST

    # evaluate the model
    rmse(x, y, beta)
    r2_score(x, y, beta)
    plot_regression_line(x, y, beta, xlabel='Head Size in cm3', ylabel='Brain Weight in grams')


if __name__ == "__main__":
    main()
