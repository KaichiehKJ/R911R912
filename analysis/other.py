import pandas as pd
from sklearn.linear_model import LinearRegression
from sympy import *

data = pd.read_csv("result/data/process_ARO2-DCS-PDI91101_data.csv")
# print(data.columns)
data = data.dropna()


def R911_linear(FI91601, LIMS_S905_Water, LIMS_S907_Water, FA, L2, PDI91101):

    slope = [0.005229192897741963, -0.004587247410881216, 0.00038968913035516517, -5.925344380082161e-05, -6.991390418588193, 0.5144619628338897]

    intercept = 5.908770431056638

    FI91601_slope = (slope[1] * FI91601)
    LIMS_S905_Water_slope = (slope[2] * LIMS_S905_Water)
    LIMS_S907_Water_slope = (slope[3] * LIMS_S907_Water)
    FA_slope = (slope[4] * FA)
    L2_slope = (slope[5] * L2)

    time = (PDI91101 - FI91601_slope - LIMS_S905_Water_slope - LIMS_S907_Water_slope - FA_slope - L2_slope - intercept) / slope[0]

    return time


def R911_lasso(FI91601, LIMS_S905_Water, LIMS_S907_Water, FA, L2, PDI91101):

    slope = [0.0021093666237582483, -0.0046808487586839495, 0.0007693930672083095, 0.0, 0.0, 0.0]
    intercept = 2.718415037148679


    FI91601_slope = (slope[1] * FI91601)
    LIMS_S905_Water_slope = (slope[2] * LIMS_S905_Water)
    LIMS_S907_Water_slope = (slope[3] * LIMS_S907_Water)
    FA_slope = (slope[4] * FA)
    L2_slope = (slope[5] * L2)

    time = (PDI91101 - FI91601_slope - LIMS_S905_Water_slope - LIMS_S907_Water_slope - FA_slope - L2_slope - intercept) / slope[0]

    return time


def R911_polynomial(FI91601, LIMS_S905_Water, LIMS_S907_Water, FA, L2, PDI91101):
    # ['1',
    # 'time', 'FI91601', 'LIMS_S905_Water', 'LIMS_S907_Water', 'FA', 'L2',
    # 'time^2', 'time FI91601', 'time LIMS_S905_Water', 'time LIMS_S907_Water', 'time FA', 'time L2',
    # 'FI91601^2', 'FI91601 LIMS_S905_Water', 'FI91601 LIMS_S907_Water', 'FI91601 FA', 'FI91601 L2',
    # 'LIMS_S905_Water^2', 'LIMS_S905_Water LIMS_S907_Water', 'LIMS_S905_Water FA', 'LIMS_S905_Water L2',
    # 'LIMS_S907_Water^2', 'LIMS_S907_Water FA', 'LIMS_S907_Water L2',
    # 'FA^2', 'FA L2',
    # 'L2^2']

    slope = [-1.2314449838006335e-06, -0.8604898105439748, 0.1272446270371889, 0.004931419806020324, -0.018779222349636977, -109.0421024378612,
             59.275448847504286, -7.546366364602833e-05, 0.0007647865291576968, -0.0001541878934329102, -0.00022387227435804796, 1.1959176922353194,
             -0.10552121014096606, -6.474489461250066e-05, -1.0837987717325984e-05, -6.326442491266793e-07, -0.13590556041564741, 0.0068982277565869535,
             1.916077806766482e-06, -5.858395226631882e-07, -1.1093242246236251e-05, 0.0033846490363992343, 9.266167506178594e-07, 0.04170217468196517,
             -0.0018604152407673091, 220.85180467735515, -127.65130490154552, -1.34798467130059]


    intercept = -13.433861101433365

    time = symbols("time")
    x = 1 * slope[0] + FI91601 * slope[2] + LIMS_S905_Water * slope[3] + LIMS_S907_Water * slope[4] + FA * \
        slope[5] + L2 * slope[6]
    y = (FI91601 ** 2) * slope[13] + FI91601 * LIMS_S905_Water * slope[14] + FI91601 * LIMS_S907_Water * \
        slope[15] + FI91601 * FA * slope[16] + FI91601 * L2 * slope[17]
    z = (LIMS_S905_Water ** 2) * slope[18] + LIMS_S905_Water * LIMS_S907_Water * slope[
        19] + LIMS_S905_Water * FA * slope[20] + LIMS_S905_Water * L2 * slope[21]
    i = (LIMS_S907_Water ** 2) + slope[22] + LIMS_S907_Water * FA * slope[23] + LIMS_S907_Water * L2 * \
        slope[24]
    j = (FA ** 2) * slope[27] + FA * L2 * slope[26] + (L2 ** 2) * slope[25]


    content = x + y + z + i + j + intercept - PDI91101
    result = solve(time * slope[1] + (time ** 2) * slope[7] + time * FI91601 * slope[8] + time * LIMS_S905_Water *
                   slope[9] + time * LIMS_S907_Water * slope[10] + time * FA * slope[11] + time * L2 * slope[12] + content)

    result = max(result)

    return result



def R912_linear(FI91601, LIMS_S905_Water, LIMS_S907_Water, FA, L2, PDI91201):

    slope = [0.07508325216364509, -0.0060749133021643, 0.0002427952455986681, -4.5147242868843926e-05, -8.043781288008397, 0.25187819440019654]

    intercept = 7.210074563730016

    FI91601_slope = (slope[1] * FI91601)
    LIMS_S905_Water_slope = (slope[2] * LIMS_S905_Water)
    LIMS_S907_Water_slope = (slope[3] * LIMS_S907_Water)
    FA_slope = (slope[4] * FA)
    L2_slope = (slope[5] * L2)

    time = (PDI91201 - FI91601_slope - LIMS_S905_Water_slope - LIMS_S907_Water_slope - FA_slope - L2_slope - intercept) / slope[0]

    return time


def R912_lasso(FI91601, LIMS_S905_Water, LIMS_S907_Water, FA, L2, PDI91201):

    slope = [0.04273774976266989, -0.005934159883271034, 0.0008053646962900877, 4.095086473604449e-06, 0.0, 0.0]
    intercept = 3.3539380147644327


    FI91601_slope = (slope[1] * FI91601)
    LIMS_S905_Water_slope = (slope[2] * LIMS_S905_Water)
    LIMS_S907_Water_slope = (slope[3] * LIMS_S907_Water)
    FA_slope = (slope[4] * FA)
    L2_slope = (slope[5] * L2)

    time = (PDI91201 - FI91601_slope - LIMS_S905_Water_slope - LIMS_S907_Water_slope - FA_slope - L2_slope - intercept) / slope[0]

    return time


def R912_polynomial(FI91601, LIMS_S905_Water, LIMS_S907_Water, FA, L2, PDI91201):

    slope = [2.133828637869137e-07, -0.044611850885293784, 0.16514249182858434, -0.006584167226448323, -0.017056617185215152, -195.8456217115498,
             54.90287002979531, -0.0029198550015511666, 0.0004451770188908604, -0.00013809261435048369, -9.568447288297165e-05, 0.017427558006359527,
             -0.2789892090190089, -8.394564948167177e-05, 2.6196361253838507e-07, -6.580132980552739e-06, -0.18352232682848946, 0.021366470086267856,
             2.6727094732911013e-06, -2.848568393775796e-07, 0.011289749699489125, 0.002372225081751256, 3.2809415621847015e-06, 0.04444787716557307,
             -0.00341709916985012, 343.0968571381942, -151.12354154232403, 10.782574766572438]

    intercept = -1.115715609944345

    time = symbols("time")
    x = 1 * slope[0] + FI91601 * slope[2] + LIMS_S905_Water * slope[3] + LIMS_S907_Water * slope[4] + FA * \
        slope[5] + L2 * slope[6]
    y = (FI91601 ** 2) * slope[13] + FI91601 * LIMS_S905_Water * slope[14] + FI91601 * LIMS_S907_Water * \
        slope[15] + FI91601 * FA * slope[16] + FI91601 * L2 * slope[17]
    z = (LIMS_S905_Water ** 2) * slope[18] + LIMS_S905_Water * LIMS_S907_Water * slope[
        19] + LIMS_S905_Water * FA * slope[20] + LIMS_S905_Water * L2 * slope[21]
    i = (LIMS_S907_Water ** 2) + slope[22] + LIMS_S907_Water * FA * slope[23] + LIMS_S907_Water * L2 * \
        slope[24]
    j = (FA ** 2) * slope[27] + FA * L2 * slope[26] + (L2 ** 2) * slope[25]

    content = x + y + z + i + j + intercept - PDI91201

    result = solve(time * slope[1] + (time ** 2) * slope[7] + time * FI91601 * slope[8] + time * LIMS_S905_Water * slope[9] +
                   time * LIMS_S907_Water * slope[10] + time * FA * slope[11] + time * L2 * slope[12] + content)


    result = max(result)

    return result



def MX_linear(FI91601, LIMS_S708_Br_Index, LIMS_S708_A9, LIMS_S708_A10, LIMS_S708_Water, LIMS_S708_Sulfur,
              LIMS_s919_A9, LIMS_s919_A10, LIMS_S905_Water, LIMS_S907_Water, FA, L2, mx):
    slope = [0.5026537152494234, -0.1419981918239352, 0.316846141819883, -174.93106101283536, -30.25005372968611,
             1.6567280752173563e-06, 2.574886695838782, 0.01024542584819767, 0.0014449728612576096, 0.012527214098019598,
             0.056843854473878845, -171.28765027686828, -112.80014997254257]

    intercept = 203.0815902570037

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
    slope = [0.5329721785420252, -0.1200195308700459, 0.3162080819637333, 0.0, 0.0,
             4.827238991004334e-05, 0.0, 0.01016124930197615, 0.0050229248265834135,
             0.027421442466354105, 0.08645369662933292, 0.0, 0.0]

    intercept = 48.301960786461834

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


def MX_polynomial(FI91601, LIMS_S708_Br_Index, LIMS_S708_A9, LIMS_S708_A10, LIMS_S708_Water, LIMS_S708_Sulfur,
                  LIMS_s919_A9, LIMS_s919_A10, LIMS_S905_Water, LIMS_S907_Water, FA, L2, mx):
    slope = [0.5026537152494067, -0.14199819182393636, 0.31684614181929305, -174.9310610125306,
             -30.250053729715844, 1.6567280624497915e-06, 2.5748866958402066, 0.010245425848243665,
             0.0014449728612592194, 0.012527214097986341, 0.05684385447392647, -171.28765027686774, -112.80014997254285]

    intercept = 203.08159025698615

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

# vairable = ['sum_ARO2-LIMS-s922@MX', 'sum_time',
#             'mean_ARO2-DCS-FI91601', 'mean_ARO2-LIMS-S708@Br.Index',
#             'mean_ARO2-LIMS-S708@A9', 'mean_ARO2-LIMS-S708@A10+',
#             'mean_ARO2-LIMS-S708@Water', 'mean_ARO2-LIMS-S708@Sulfur',
#             'mean_ARO2-LIMS-s919@A9', 'mean_ARO2-LIMS-s919@A10+',
#             'mean_ARO2-LIMS-S905@Water', 'mean_ARO2-LIMS-S907@Water',
#             'mean_ARO2-DCS-R911_2_A_FA', 'mean_ARO2-DCS-R911_2_L2_A']


vairable = ['sum_ARO2-DCS-PDI91101', 'sum_time',
            "mean_ARO2-DCS-FI91601", "mean_ARO2-LIMS-S905@Water",
            "mean_ARO2-LIMS-S907@Water", "mean_ARO2-DCS-R911_2_A_FA",
            "mean_ARO2-DCS-R911_2_L2_A"]


time_result = []
# for items in data[vairable].values:
#      time_result.append(MX_linear(FI91601 = items[2], LIMS_S708_Br_Index = items[3], LIMS_S708_A9 = items[4], LIMS_S708_A10 = items[5], LIMS_S708_Water = items[6],
#                                  LIMS_S708_Sulfur = items[7], LIMS_s919_A9 = items[8], LIMS_s919_A10 = items[9], LIMS_S905_Water = items[10],
#                                  LIMS_S907_Water = items[11], FA = items[12], L2 = items[13], mx = items[0]))

for items in data[vairable].values:
    time_result.append(R911_polynomial(FI91601 = items[2], LIMS_S905_Water = items[3], LIMS_S907_Water = items[4], FA = items[5], L2 = items[6], PDI91101 = items[0]))

model = LinearRegression()
model.fit(pd.DataFrame(time_result), data[["sum_time"]])
model_info = [model.coef_, model.intercept_]
print(model_info)