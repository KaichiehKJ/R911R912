
from utils.process import ProcessData
from analysis.time_linear import TimeLinear
import os


if __name__=="__main__":

    path = "data/R911R912 _明志蔡教授_R4-ARO2.xlsx"
    target = ["ARO2-LIMS-s922@MX"]
    time = ["time"]
    variable = ["ARO2-DCS-FI91601", "ARO2-LIMS-S708@Br.Index", "ARO2-LIMS-S708@A9", "ARO2-LIMS-S708@A10+",
                "ARO2-LIMS-S708@Water", "ARO2-LIMS-S708@Sulfur", "ARO2-LIMS-s919@A9", "ARO2-LIMS-s919@A10+",
                "ARO2-LIMS-S905@Water", "ARO2-LIMS-S907@Water", "ARO2-DCS-R911_2_A_FA", "ARO2-DCS-R911_2_L2_A"]

    remove_FI91601_value = 520  # 520, 470
    method = "pd_fill"  # pd_fill, knn
    alpha = 0.1
    degree = 1
    methods = ["linear_regression", "lasso", "polynomial"]


    if "process_{target}_data.csv".format(target = target[0]) not in os.listdir("result/data/"):
        process_data = ProcessData(path = path, target = target, time = time, variable = variable)
        process_data.data_cleansing(remove_FI91601_value = remove_FI91601_value)
        process_data.fill_value(method=method)
        process_data.calculate_target_time_variable()
        process_data.get_df()
    else:
        time_linear = TimeLinear(target = target, time = time, variable = variable)
        time_linear.pre_process()
        time_linear.analysis(methods = methods, alpha = alpha, degree = degree)