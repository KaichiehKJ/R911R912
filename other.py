import pandas as pd
from sklearn.linear_model import LinearRegression
from sympy import *

data = pd.read_csv("result/data/process_ARO2-LIMS-s922@MX_data.csv")
# print(data.columns)
data = data.dropna()


def MX_linear(FI91601, LIMS_S708_Br_Index, LIMS_S708_A9, LIMS_S708_A10, LIMS_S708_Water, LIMS_S708_Sulfur,
              LIMS_s919_A9, LIMS_s919_A10, LIMS_S905_Water, LIMS_S907_Water, FA, L2, mx):
    slope = [0.13898741991146374, -0.2192520032171638, 0.1169036498830991, -67.54151789158917, 863.5613072310462,
             0.00020593865918475465, 1.3119128229955244,
             0.011964330222391285, 0.011964330222391285, 0.02127341662049617, -0.02440569498166165, -82.28030073983571,
             -43.82636991857343]
    intercept = 179.34789191010213

    FI91601 = (slope[1] * FI91601)
    LIMS_S708_Br_Index = (slope[2] * LIMS_S708_Br_Index)
    LIMS_S708_A9 = (slope[3] * LIMS_S708_A9)
    LIMS_S708_A10 = (slope[4] * LIMS_S708_A10)
    LIMS_S708_Water = (slope[5] * LIMS_S708_Water)
    LIMS_S708_Sulfur = (slope[6] * LIMS_S708_Sulfur)
    LIMS_s919_A9 = (slope[7] * LIMS_s919_A9)
    LIMS_s919_A10 = (slope[8] * LIMS_s919_A10)
    LIMS_S905_Water = (slope[9] * LIMS_S905_Water)
    LIMS_S907_Water = (slope[10] * LIMS_S907_Water)
    FA = (slope[11] * FA)
    L2 = (slope[12] * L2)

    time = (mx - FI91601 - LIMS_S708_Br_Index - LIMS_S708_A9 - LIMS_S708_A10 - LIMS_S708_Water - LIMS_S708_Sulfur - LIMS_s919_A9 - LIMS_s919_A10 - LIMS_S905_Water - LIMS_S907_Water - FA - L2 - intercept) / slope[0]

    return time


def MX_lasso(FI91601, LIMS_S708_Br_Index, LIMS_S708_A9, LIMS_S708_A10, LIMS_S708_Water, LIMS_S708_Sulfur,
             LIMS_s919_A9, LIMS_s919_A10, LIMS_S905_Water, LIMS_S907_Water, FA, L2, mx):
    slope = [0.14195629704252172, -0.23086430951083733, 0.10713970768673176, 0.0, 0.0, 0.00023792560254264967, 0.0,
             0.011651668559803069,
             0.004932498816112235, 0.032950414212017984, -0.015278892382212821, 0.0, 0.0]
    intercept = 121.34207314394506

    FI91601 = (slope[1] * FI91601)
    LIMS_S708_Br_Index = (slope[2] * LIMS_S708_Br_Index)
    LIMS_S708_A9 = (slope[3] * LIMS_S708_A9)
    LIMS_S708_A10 = (slope[4] * LIMS_S708_A10)
    LIMS_S708_Water = (slope[5] * LIMS_S708_Water)
    LIMS_S708_Sulfur = (slope[6] * LIMS_S708_Sulfur)
    LIMS_s919_A9 = (slope[7] * LIMS_s919_A9)
    LIMS_s919_A10 = (slope[8] * LIMS_s919_A10)
    LIMS_S905_Water = (slope[9] * LIMS_S905_Water)
    LIMS_S907_Water = (slope[10] * LIMS_S907_Water)
    FA = (slope[11] * FA)
    L2 = (slope[12] * L2)

    time = (
                       mx - FI91601 - LIMS_S708_Br_Index - LIMS_S708_A9 - LIMS_S708_A10 - LIMS_S708_Water - LIMS_S708_Sulfur - LIMS_s919_A9 - LIMS_s919_A10 - LIMS_S905_Water - LIMS_S907_Water - FA - L2 - intercept) / \
           slope[0]

    return time


def MX_polynomial(FI91601, LIMS_S708_Br_Index, LIMS_S708_A9, LIMS_S708_A10, LIMS_S708_Water, LIMS_S708_Sulfur,
                  LIMS_s919_A9, LIMS_s919_A10, LIMS_S905_Water, LIMS_S907_Water, FA, L2, mx):
    slope = [0.13898741991150665, -0.21925200321690977, 0.11690364988336008, -67.54151789130312, 863.5613072301774,
             0.00020593865917288914, 1.3119128230103796, 0.011964330222421979,
             0.0032547311375946686, 0.021273416620704463, -0.02440569498166398, -82.28030073983697, -43.82636991857313]
    intercept = 179.34789190992288

    FI91601 = (slope[1] * FI91601)
    LIMS_S708_Br_Index = (slope[2] * LIMS_S708_Br_Index)
    LIMS_S708_A9 = (slope[3] * LIMS_S708_A9)
    LIMS_S708_A10 = (slope[4] * LIMS_S708_A10)
    LIMS_S708_Water = (slope[5] * LIMS_S708_Water)
    LIMS_S708_Sulfur = (slope[6] * LIMS_S708_Sulfur)
    LIMS_s919_A9 = (slope[7] * LIMS_s919_A9)
    LIMS_s919_A10 = (slope[8] * LIMS_s919_A10)
    LIMS_S905_Water = (slope[9] * LIMS_S905_Water)
    LIMS_S907_Water = (slope[10] * LIMS_S907_Water)
    FA = (slope[11] * FA)
    L2 = (slope[12] * L2)

    time = (
                       mx - FI91601 - LIMS_S708_Br_Index - LIMS_S708_A9 - LIMS_S708_A10 - LIMS_S708_Water - LIMS_S708_Sulfur - LIMS_s919_A9 - LIMS_s919_A10 - LIMS_S905_Water - LIMS_S907_Water - FA - L2 - intercept) / \
           slope[0]

    return time

vairable = ['sum_ARO2-LIMS-s922@MX', 'sum_time',
       'mean_ARO2-DCS-FI91601', 'mean_ARO2-LIMS-S708@Br.Index',
       'mean_ARO2-LIMS-S708@A9', 'mean_ARO2-LIMS-S708@A10+',
       'mean_ARO2-LIMS-S708@Water', 'mean_ARO2-LIMS-S708@Sulfur',
       'mean_ARO2-LIMS-s919@A9', 'mean_ARO2-LIMS-s919@A10+',
       'mean_ARO2-LIMS-S905@Water', 'mean_ARO2-LIMS-S907@Water',
       'mean_ARO2-DCS-R911_2_A_FA', 'mean_ARO2-DCS-R911_2_L2_A']
time_result = []
for items in data[vairable].values:
     time_result.append(MX_polynomial(FI91601 = items[2], LIMS_S708_Br_Index = items[3], LIMS_S708_A9 = items[4], LIMS_S708_A10 = items[5], LIMS_S708_Water = items[6],
                                  LIMS_S708_Sulfur = items[7], LIMS_s919_A9 = items[8], LIMS_s919_A10 = items[9], LIMS_S905_Water = items[10],
                                  LIMS_S907_Water = items[11], FA = items[12], L2 = items[13], mx = items[0]))

model = LinearRegression()
model.fit(pd.DataFrame(time_result), data[["sum_time"]])
model_info = [model.coef_, model.intercept_]
print(model_info)