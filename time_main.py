
from utils.process import ProcessData
from analysis.time_linear import TimeLinear
import os


if __name__=="__main__":

    """
    target = ["ARO2-LIMS-s922@MX"]
    variable
    ["ARO2-DCS-FI91601", "ARO2-LIMS-S708@Br.Index", "ARO2-LIMS-S708@A9", "ARO2-LIMS-S708@A10+",
                "ARO2-LIMS-S708@Water", "ARO2-LIMS-S708@Sulfur", "ARO2-LIMS-s919@A9", "ARO2-LIMS-s919@A10+",
                "ARO2-LIMS-S905@Water", "ARO2-LIMS-S907@Water", "ARO2-DCS-R911_2_A_FA", "ARO2-DCS-R911_2_L2_A"]
    modify = 200 
    
    target = ["ARO2-DCS-PDI91101"]
    variable
    ["ARO2-DCS-FI91601", "ARO2-DCS-FI91701", "ARO2-LIMS-S905@Water", "ARO2-LIMS-S907@Water", "ARO2-DCS-R911_2_A_FA", "ARO2-DCS-R911_2_L2_A"]
    modify = None
    
    target = ["ARO2-DCS-PDI91201"]
    variable
    ["ARO2-DCS-FI91601", "ARO2-DCS-FI91701", "ARO2-LIMS-S905@Water", "ARO2-LIMS-S907@Water", "ARO2-DCS-R911_2_A_FA", "ARO2-DCS-R911_2_L2_A"]
    modify = None
    """

    path = "data/R911R912 _明志蔡教授_R4-ARO2.xlsx"
    target = ["ARO2-LIMS-s922@MX"]
    time = ["time"]
    variable = ["ARO2-DCS-FI91601", "ARO2-LIMS-S708@Br.Index", "ARO2-LIMS-S708@A9", "ARO2-LIMS-S708@A10+",
                "ARO2-LIMS-S708@Water", "ARO2-LIMS-S708@Sulfur", "ARO2-LIMS-s919@A9", "ARO2-LIMS-s919@A10+",
                "ARO2-LIMS-S905@Water", "ARO2-LIMS-S907@Water", "ARO2-DCS-R911_2_A_FA", "ARO2-DCS-R911_2_L2_A"]
    modify = 200                # None

    update = False

    remove_FI91601_value = 520  # 520, 470
    method = "pd_fill"          # pd_fill, knn
    alpha = 0.1
    degree = 1
    methods = ["linear_regression", "lasso", "polynomial"]


    if ("process_{target}_data.csv".format(target = target[0]) not in os.listdir("result/data/")) | (update):
        process_data = ProcessData(path = path, target = target, time = time, variable = variable)
        process_data.data_cleansing(remove_FI91601_value = remove_FI91601_value)
        process_data.fill_value(method=method)
        process_data.calculate_target_time_variable()
        process_data.write_csv()
    else:
        time_linear = TimeLinear(target = target, time = time, variable = variable, modify = modify)
        time_linear.pre_process()
        time_linear.analysis(methods = methods, alpha = alpha, degree = degree)