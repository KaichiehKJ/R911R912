
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


class TimeLinear():

    def __init__(self, target, time, variable, modify):

        self._target = target
        self._time = time
        self._variable = variable
        self._modify = modify
        self._set_analyzer_variable()
        self._read_process_data()


    def _set_analyzer_variable(self):

        self._sum_target = ["sum_" + self._target[0]]
        self._sum_time = ["sum_" + self._time[0]]
        self._mean_variable = ["mean_" + variable for variable in self._variable]


    def _read_process_data(self):

        path = "result/data/process_{target}_data.csv".format(target=self._target[0])
        self.data = pd.read_csv(path)


    def _select_data(self):

        self.liner_df = self.data[self._sum_time + self._mean_variable + self._sum_target]

        if (self._modify is not None) & (self._target[0] == "ARO2-LIMS-s922@MX"):
            self.liner_df = self.liner_df.loc[self.liner_df[self._sum_target[0]] < self._modify]

        self.liner_df = self.liner_df.dropna()


    def _split_data(self):
        # splitting the data

        self._x_train, self._x_test, self._y_train, self._y_test = train_test_split(self.liner_df[self._sum_time + self._mean_variable],
                                                                                    self.liner_df[self._sum_target[0]], test_size = 0.2,
                                                                                    random_state = 42)

    def pre_process(self):

        self._select_data()
        self._split_data()


    def _evaluate(self, y_prediction):

        r2 = r2_score(self._y_test, y_prediction)
        mse = mean_squared_error(self._y_test, y_prediction)
        rmse = np.sqrt(mean_squared_error(self._y_test, y_prediction))

        return [r2, mse, rmse]


    def _write_text(self, method, score, model_info, degree):

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        path = "result/{target}_result.txt".format(target = self._target[0])
        r2_text = "r2 = {r2}\n".format(r2 = score[0])
        mse_text = "MSE = {mse}\n".format(mse = score[1])
        rmse_text = "RMSE = {rmse}\n".format(rmse = score[2])
        slope_text = "slope : {slope}\n".format(slope = model_info[0].tolist())
        intercept_text = "intercept : {intercept}\n".format(intercept = model_info[1].tolist())
        degree_text = "degree :{degree}\n".format(degree = degree)

        with open(path, 'a') as f:
            f.write(current_time + "\n")
            f.write(method + ":\n")
            if method == "polynomial":
                f.write(degree_text)
            f.write(r2_text)
            f.write(mse_text)
            f.write(rmse_text)
            f.write(slope_text)
            f.write(intercept_text)
            f.write("\n")


    def analysis(self, methods, alpha, degree):

        for method in methods:
            if method == "linear_regression":
                model = LinearRegression()
            elif method == "lasso":
                model = Lasso(alpha = alpha)
            elif method == "polynomial":
                model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

            model.fit(self._x_train, self._y_train)
            y_prediction = model.predict(self._x_test)
            score = self._evaluate(y_prediction = y_prediction)
            if method == "polynomial":
                model_info = [model.steps[1][1].coef_, model.steps[1][1].intercept_]
            else:
                model_info = [model.coef_, model.intercept_]

            self._write_text(method, score, model_info, degree)
