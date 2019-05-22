import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from utils import plot_regression_line
import os


def main():
    # read data from csv file
    data = pd.read_csv(os.path.abspath('headbrain.csv'))
    print("data.shape = {}".format(data.shape))

    # load data to x and y
    X = data['Head Size(cm^3)'].values
    X = X.reshape(X.shape[0], 1)
    y = data['Brain Weight(grams)'].values

    # creating model
    model = LinearRegression()

    # fitting training data
    model.fit(X, y)

    # y prediction
    y_pred = model.predict(X)

    # coefficients
    print('Coefficients: beta0 = {}, beta1 = {}'.format(model.intercept_, model.coef_))

    # calculating RMSE and R2 score
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2_score = model.score(X, y)
    print('RMSE = {}'.format(rmse))
    print('R2_Score = {}'.format(r2_score))

    # TEST
    test_pred = model.predict(np.matrix([3000]))
    print("Predicted value = {}".format(test_pred))
    # END TEST

    plot_regression_line(X, y, [model.intercept_, model.coef_], xlabel='Head Size in cm3',
                         ylabel='Brain Weight in grams')


if __name__ == "__main__":
    main()
