
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.impute import KNNImputer
from datetime import datetime

class ProcessData():

    def __init__(self, path, target, time, variable, method, remove_FI91601_value):

        self._target = target
        self._time = time
        self._variable = variable
        self._remove_FI91601_value = remove_FI91601_value
        self._set_normal_time()
        self._read_xlsx(path = path)
        self._select_variable()
        self.data_cleansing()
        self.fill_value(method=method)
        self._select_normal_time()

    def _set_normal_time(self):

        self._normal_time_df = pd.read_csv("config/normal_operation_time.csv")

        self._normal_time_df["start_time"] = pd.to_datetime(self._normal_time_df["start_time"], format="%Y%m%d%H%M", errors='ignore')
        self._normal_time_df["end_time"] = pd.to_datetime(self._normal_time_df["end_time"], format="%Y%m%d%H%M", errors='ignore')


    def _read_xlsx(self, path):

        self.df_x = pd.read_excel(path, sheet_name = "R911R912_TAG相關錶點資料(X)")
        self.df_y = pd.read_excel(path, sheet_name = "目標值(Y)相關錶點資料")

        # self.df_x["Unnamed: 0"] = range(len(self.df_x["Unnamed: 0"]))
        # self.df_x = self.df_x.rename(columns={"Unnamed: 0":self._time[0]})


    def _select_variable(self):

        if self._target[0] in self.df_x.columns:
            # self.df = self.df_x[self._time + self._variable + self._target]
            self.df = self.df_x[["Unnamed: 0"] + self._variable + self._target]
        else:
            # self.df = self.df_x[self._time + self._variable].join(self.df_y[self._target[0]])
            self.df = self.df_x[["Unnamed: 0"] + self._variable].join(self.df_y[self._target[0]])


    def _remove_FI91601(self, df):

        df = df.loc[df["ARO2-DCS-FI91601"] > self._remove_FI91601_value, :]
        df.reset_index(inplace=True, drop=True)

        return df


    def _select_normal_time(self):

        self._normal_df_list = []
        for index in range(len(self._normal_time_df)):
            start_time, end_time = self._normal_time_df.loc[index,:]
            normal_df = self.df.loc[(self.df["Unnamed: 0"] >= start_time) & (self.df["Unnamed: 0"] < end_time),:]
            if len(normal_df) > 0:
                normal_df[self._time[0]] = range(len(normal_df))

                for column in normal_df.columns[1:]:
                    normal_df[column] = pd.to_numeric(normal_df[column], errors='coerce')

                normal_df = self._remove_FI91601(df = normal_df)

                self._normal_df_list.append(normal_df)


    def _process_str_in_cell(self):

        for column in self.df.columns:
            self.df.loc[self.df[column] == "Over Range", column] = np.nan
            # self.df[column] = pd.to_numeric(self.df[column], errors='coerce')


    def _remove_bad_data(self):

        """
        remove:
        ARO2-DCS-FI91601 > 520 or 470
        R911 & R912 > 4
        MX > 1700
        """

        # self.df = self.df.loc[self.df["ARO2-DCS-FI91601"] > remove_FI91601_value,:]
        if self._target[0] == "ARO2-LIMS-s922@MX":
            self.df.loc[self.df[self._target[0]] > 1700, self._target[0]] = np.nan
        else:
            self.df[self._target[0]] = pd.to_numeric(self.df[self._target[0]], errors='coerce')
            self.df.loc[self.df[self._target[0]] > 4, self._target[0]] = np.nan

        # self.df.reset_index(inplace=True, drop=True)


    def data_cleansing(self):

        self._process_str_in_cell()
        self._remove_bad_data()


    def _pd_fill_method(self):

        for variable in self._variable:
            self.df[variable].fillna(method='ffill', inplace = True)
            self.df[variable].fillna(method='bfill', inplace = True)


    def _knn_method(self):

        imputer = KNNImputer(n_neighbors=3)
        imputed = imputer.fit_transform(self.df[self._time + self._variable])
        df_imputed = pd.DataFrame(imputed, columns = self._time + self._variable)

        self.df = df_imputed.join(self.df[self._target[0]])


    def fill_value(self, method):

        # self._process_str_in_cell()
        if method == "knn":
            self._knn_method()
        elif method == "pd_fill":
            self._pd_fill_method()


    def calculate_btw_time_and_traget(self, df, index_list):

        for item in self._time + self._target:
            for index in range(len(index_list) - 1):
                df.loc[index_list[index + 1], "btw_" + item] =  df.loc[index_list[index + 1], item] - df.loc[index_list[index], item]

        return df

    def calculate_target_time_variable(self):

        for df in self._normal_df_list:
            index_list = df[pd.notna(df[self._target[0]])].index
            df = self.calculate_btw_time_and_traget(df = df, index_list = index_list)
            n = 0
            for index in range(len(index_list)):
                if df.loc[index_list[index], "btw_" + self._target[0]] > 0:
                    if index_list[index] == index_list[len(index_list)-1]:
                        df.loc[index_list[index], "sum_" + self._target[0]] = df.loc[index_list[index-n]:, "btw_" + self._target[0]].sum()
                        df.loc[index_list[index], "sum_" + self._time[0]] = df.loc[index_list[index-n]:, "btw_" + self._time[0]].sum()
                        for variable in self._variable:
                            df.loc[index_list[index], "mean_" + variable] = df.loc[(index_list[index-n-1]+1):, variable].mean()
                    else:
                        n = n + 1
                else:
                    if index_list[index] != index_list[0]:
                        if n != 0:
                            df.loc[index_list[index-1], "sum_" + self._target[0]] = df.loc[index_list[index-n]:index_list[index-1], "btw_" + self._target[0]].sum()
                            df.loc[index_list[index-1], "sum_" + self._time[0]] = df.loc[index_list[index-n]:index_list[index-1], "btw_" + self._time[0]].sum()
                            for variable in self._variable:
                                df.loc[index_list[index-1], "mean_" + variable] = df.loc[(index_list[index-n-1]+1):index_list[index-1], variable].mean()

                    n = 0


    def write_csv(self):

        path = "result/data/process_{target}_data.csv".format(target = self._target[0])
        self.new_df = pd.DataFrame()
        for df in self._normal_df_list:
            self.new_df = pd.concat([self.new_df, df])

        self.new_df.to_csv(path)


    def get_df(self):

        return self.new_df


if __name__ == "__main__":

    path = "data/R911R912 _明志蔡教授_R4-ARO2.xlsx"
    target = ["ARO2-LIMS-s922@MX"]
    time = ["time"]
    variable = ["ARO2-DCS-FI91601", "ARO2-LIMS-S708@Br.Index", "ARO2-LIMS-S708@A9", "ARO2-LIMS-S708@A10+",
                "ARO2-LIMS-S708@Water", "ARO2-LIMS-S708@Sulfur", "ARO2-LIMS-s919@A9", "ARO2-LIMS-s919@A10+",
                "ARO2-LIMS-S905@Water", "ARO2-LIMS-S907@Water", "ARO2-DCS-R911_2_A_FA", "ARO2-DCS-R911_2_L2_A"]

    remove_FI91601_value = 520
    method = "pd_fill"
    process_data = ProcessData(path = path, target = target, time = time, variable = variable)
    process_data.data_cleansing(remove_FI91601_value = remove_FI91601_value)
    process_data.fill_value(method = method)
    process_data.calculate_btw_time_and_traget()
    process_data.name()
