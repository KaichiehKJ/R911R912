
from sympy import *
import pandas as pd

def R911_linear(FI91601_list, LIMS_S905_Water, LIMS_S907_Water, FA_list, L2_list, PDI91101):

    slope = [0.005229192897741963, -0.004587247410881216, 0.00038968913035516517, -5.925344380082161e-05,
             -6.991390418588193, 0.5144619628338897]

    intercept = 5.908770431056638

    result_dict = {
        "method":[],
        "time":[],
        "PDI91101":[],
        "FI91601":[],
        "FA":[],
        "L2":[],
        "LIMS_S905_Water":[],
        "LIMS_S907_Water":[]
    }

    for FI91601 in FI91601_list:
        for FA in FA_list:
            for L2 in L2_list:

                FI91601_slope = (slope[1] * FI91601)
                LIMS_S905_Water_slope = (slope[2] * LIMS_S905_Water)
                LIMS_S907_Water_slope = (slope[3] * LIMS_S907_Water)
                FA_slope = (slope[4] * FA)
                L2_slope = (slope[5] * L2)

                time = (PDI91101 - FI91601_slope - LIMS_S905_Water_slope - LIMS_S907_Water_slope - FA_slope - L2_slope - intercept) / slope[0]
                time = round(0.03428085 * time + 2.19749761, 2)

                result_dict["method"].append("Linear")
                result_dict["time"].append(time)
                result_dict["PDI91101"].append(PDI91101)
                result_dict["FI91601"].append(FI91601)
                result_dict["FA"].append(FA)
                result_dict["L2"].append(L2)
                result_dict["LIMS_S905_Water"].append(LIMS_S905_Water)
                result_dict["LIMS_S907_Water"].append(LIMS_S907_Water)

    linear_pd = pd.DataFrame(result_dict)

    return linear_pd


def R911_lasso(FI91601_list, LIMS_S905_Water, LIMS_S907_Water, FA_list, L2_list, PDI91101):

    slope = [0.0021093666237582483, -0.0046808487586839495, 0.0007693930672083095, 0.0, 0.0, 0.0]
    intercept = 2.718415037148679

    result_dict = {
        "method": [],
        "time": [],
        "PDI91101": [],
        "FI91601": [],
        "FA": [],
        "L2": [],
        "LIMS_S905_Water": [],
        "LIMS_S907_Water": []
    }

    for FI91601 in FI91601_list:
        for FA in FA_list:
            for L2 in L2_list:

                FI91601_slope = (slope[1] * FI91601)
                LIMS_S905_Water_slope = (slope[2] * LIMS_S905_Water)
                LIMS_S907_Water_slope = (slope[3] * LIMS_S907_Water)
                FA_slope = (slope[4] * FA)
                L2_slope = (slope[5] * L2)

                time = (PDI91101 - FI91601_slope - LIMS_S905_Water_slope - LIMS_S907_Water_slope - FA_slope - L2_slope - intercept) / slope[0]
                time = round(0.01298659 * time + 2.2456211, 2)

                result_dict["method"].append("Lasso")
                result_dict["time"].append(time)
                result_dict["PDI91101"].append(PDI91101)
                result_dict["FI91601"].append(FI91601)
                result_dict["FA"].append(FA)
                result_dict["L2"].append(L2)
                result_dict["LIMS_S905_Water"].append(LIMS_S905_Water)
                result_dict["LIMS_S907_Water"].append(LIMS_S907_Water)

    lasso_pd = pd.DataFrame(result_dict)

    return lasso_pd


def R911_polynomial(FI91601_list, LIMS_S905_Water, LIMS_S907_Water, FA_list, L2_list, PDI91101):
    # ['1', 
    # 'time', 'FI91601', 'LIMS_S905_Water', 'LIMS_S907_Water', 'FA', 'L2', 
    # 'time^2', 'time FI91601', 'time LIMS_S905_Water', 'time LIMS_S907_Water', 'time FA', 'time L2',
    # 'FI91601^2', 'FI91601 LIMS_S905_Water', 'FI91601 LIMS_S907_Water', 'FI91601 FA', 'FI91601 L2', 
    # 'LIMS_S905_Water^2', 'LIMS_S905_Water LIMS_S907_Water', 'LIMS_S905_Water FA', 'LIMS_S905_Water L2',
    # 'LIMS_S907_Water^2', 'LIMS_S907_Water FA', 'LIMS_S907_Water L2',
    # 'FA^2', 'FA L2',
    # 'L2^2']

    slope = [-1.2314449838006335e-06, -0.8604898105439748, 0.1272446270371889, 0.004931419806020324,
             -0.018779222349636977, -109.0421024378612,
             59.275448847504286, -7.546366364602833e-05, 0.0007647865291576968, -0.0001541878934329102,
             -0.00022387227435804796, 1.1959176922353194,
             -0.10552121014096606, -6.474489461250066e-05, -1.0837987717325984e-05, -6.326442491266793e-07,
             -0.13590556041564741, 0.0068982277565869535,
             1.916077806766482e-06, -5.858395226631882e-07, -1.1093242246236251e-05, 0.0033846490363992343,
             9.266167506178594e-07, 0.04170217468196517,
             -0.0018604152407673091, 220.85180467735515, -127.65130490154552, -1.34798467130059]

    result_dict = {
        "method": [],
        "time": [],
        "PDI91101": [],
        "FI91601": [],
        "FA": [],
        "L2": [],
        "LIMS_S905_Water": [],
        "LIMS_S907_Water": []
    }

    for FI91601 in FI91601_list:
        for FA in FA_list:
            for L2 in L2_list:

                time = symbols("time")
                x = 1 * slope[0] + FI91601 * slope[2] + LIMS_S905_Water * slope[3] + LIMS_S907_Water * slope[4] + FA * slope[5] + L2 * slope[6]
                y = (FI91601**2) * slope[13] + FI91601 * LIMS_S905_Water * slope[14] + FI91601 * LIMS_S907_Water * slope[15] + FI91601 * FA * slope[16] + FI91601 * L2 * slope[17]
                z = (LIMS_S905_Water**2) * slope[18] + LIMS_S905_Water * LIMS_S907_Water * slope[19] + LIMS_S905_Water * FA * slope[20] + LIMS_S905_Water * L2 * slope[21]
                i = (LIMS_S907_Water**2) + slope[22] + LIMS_S907_Water * FA * slope[23] + LIMS_S907_Water * L2 * slope[24]
                j = (FA**2) * slope[27] + FA * L2 * slope[26] + (L2**2) * slope[25]

                intercept =  -13.433861101433365
                content = x + y + z + i + j + intercept - PDI91101
                result = solve(time * slope[1]  + (time**2) * slope[7] + time * FI91601 * slope[8] + time * LIMS_S905_Water * slope[9] + time * LIMS_S907_Water * slope[10] + time * FA * slope[11] + time * L2 * slope[12] + content)
                try:
                    if (isinstance(float(result[0]), float)) | (isinstance(float(result[1]), float)):

                        result = max(result)
                        time = round(3.46548331e-05 * result + 1.93535394, 2)

                        result_dict["method"].append("Polynomial")
                        result_dict["time"].append(time)
                        result_dict["PDI91101"].append(PDI91101)
                        result_dict["FI91601"].append(FI91601)
                        result_dict["FA"].append(FA)
                        result_dict["L2"].append(L2)
                        result_dict["LIMS_S905_Water"].append(LIMS_S905_Water)
                        result_dict["LIMS_S907_Water"].append(LIMS_S907_Water)
                except:
                    pass

    polynomial_pd = pd.DataFrame(result_dict)

    return polynomial_pd


def R912_linear(FI91601_list, LIMS_S905_Water, LIMS_S907_Water, FA_list, L2_list, PDI91201):

    slope = [0.07508325216364509, -0.0060749133021643, 0.0002427952455986681, -4.5147242868843926e-05,
             -8.043781288008397, 0.25187819440019654]

    intercept = 7.210074563730016

    result_dict = {
        "method": [],
        "time": [],
        "PDI91201": [],
        "FI91601": [],
        "FA": [],
        "L2": [],
        "LIMS_S905_Water": [],
        "LIMS_S907_Water": []
    }

    for FI91601 in FI91601_list:
        for FA in FA_list:
            for L2 in L2_list:

                FI91601_slope = (slope[1] * FI91601)
                LIMS_S905_Water_slope = (slope[2] * LIMS_S905_Water)
                LIMS_S907_Water_slope = (slope[3] * LIMS_S907_Water)
                FA_slope = (slope[4] * FA)
                L2_slope = (slope[5] * L2)

                time = (PDI91201 - FI91601_slope - LIMS_S905_Water_slope - LIMS_S907_Water_slope - FA_slope - L2_slope - intercept) / slope[0]
                time = round(0.52242928 * time + 1.17317857, 2)

                result_dict["method"].append("Linear")
                result_dict["time"].append(time)
                result_dict["PDI91201"].append(PDI91201)
                result_dict["FI91601"].append(FI91601)
                result_dict["FA"].append(FA)
                result_dict["L2"].append(L2)
                result_dict["LIMS_S905_Water"].append(LIMS_S905_Water)
                result_dict["LIMS_S907_Water"].append(LIMS_S907_Water)


    linear_pd = pd.DataFrame(result_dict)

    return linear_pd


def R912_lasso(FI91601_list, LIMS_S905_Water, LIMS_S907_Water, FA_list, L2_list, PDI91201):

    slope = [0.04273774976266989, -0.005934159883271034, 0.0008053646962900877, 4.095086473604449e-06, 0.0, 0.0]

    intercept = 3.3539380147644327

    result_dict = {
        "method": [],
        "time": [],
        "PDI91201": [],
        "FI91601": [],
        "FA": [],
        "L2": [],
        "LIMS_S905_Water": [],
        "LIMS_S907_Water": []
    }

    for FI91601 in FI91601_list:
        for FA in FA_list:
            for L2 in L2_list:
                FI91601_slope = (slope[1] * FI91601)
                LIMS_S905_Water_slope = (slope[2] * LIMS_S905_Water)
                LIMS_S907_Water_slope = (slope[3] * LIMS_S907_Water)
                FA_slope = (slope[4] * FA)
                L2_slope = (slope[5] * L2)

                time = (PDI91201 - FI91601_slope - LIMS_S905_Water_slope - LIMS_S907_Water_slope - FA_slope - L2_slope - intercept) / slope[0]
                time = round(0.27422186 * time + 1.78095009, 2)

                result_dict["method"].append("Lasso")
                result_dict["time"].append(time)
                result_dict["PDI91201"].append(PDI91201)
                result_dict["FI91601"].append(FI91601)
                result_dict["FA"].append(FA)
                result_dict["L2"].append(L2)
                result_dict["LIMS_S905_Water"].append(LIMS_S905_Water)
                result_dict["LIMS_S907_Water"].append(LIMS_S907_Water)

            lasso_pd = pd.DataFrame(result_dict)

    return lasso_pd


def R912_polynomial(FI91601_list, LIMS_S905_Water, LIMS_S907_Water, FA_list, L2_list, PDI91201):


    slope = [2.133828637869137e-07, -0.044611850885293784, 0.16514249182858434, -0.006584167226448323,
             -0.017056617185215152, -195.8456217115498,
             54.90287002979531, -0.0029198550015511666, 0.0004451770188908604, -0.00013809261435048369,
             -9.568447288297165e-05, 0.017427558006359527,
             -0.2789892090190089, -8.394564948167177e-05, 2.6196361253838507e-07, -6.580132980552739e-06,
             -0.18352232682848946, 0.021366470086267856,
             2.6727094732911013e-06, -2.848568393775796e-07, 0.011289749699489125, 0.002372225081751256,
             3.2809415621847015e-06, 0.04444787716557307,
             -0.00341709916985012, 343.0968571381942, -151.12354154232403, 10.782574766572438]

    result_dict = {
        "method": [],
        "time": [],
        "PDI91201": [],
        "FI91601": [],
        "FA": [],
        "L2": [],
        "LIMS_S905_Water": [],
        "LIMS_S907_Water": []
    }

    for FI91601 in FI91601_list:
        for FA in FA_list:
            for L2 in L2_list:
                time = symbols("time")
                x = 1 * slope[0] + FI91601 * slope[2] + LIMS_S905_Water * slope[3] + LIMS_S907_Water * slope[4] + FA * slope[5] + L2 * slope[6]
                y = (FI91601**2) * slope[13] + FI91601 * LIMS_S905_Water * slope[14] + FI91601 * LIMS_S907_Water * slope[15] + FI91601 * FA * slope[16] + FI91601 * L2 * slope[17]
                z = (LIMS_S905_Water**2) * slope[18] + LIMS_S905_Water * LIMS_S907_Water * slope[19] + LIMS_S905_Water * FA * slope[20] + LIMS_S905_Water * L2 * slope[21]
                i = (LIMS_S907_Water**2) + slope[22] + LIMS_S907_Water * FA * slope[23] + LIMS_S907_Water * L2 * slope[24]
                j = (FA**2) * slope[27] + FA * L2 * slope[26] + (L2**2) * slope[25]
                intercept =  -1.115715609944345
                content = x + y + z + i + j + intercept - PDI91201
                result = solve(time * slope[1]  + (time**2) * slope[7] + time * FI91601 * slope[8] + time * LIMS_S905_Water * slope[9] + time * LIMS_S907_Water * slope[10] + time * FA * slope[11] + time * L2 * slope[12] + content)

                try:
                    if (isinstance(float(result[0]), float)) | (isinstance(float(result[1]), float)):

                        result = max(result)
                        time = round(1.10258125e-05 * result + 2.42929156, 2)
                        result_dict["method"].append("Polynomial")
                        result_dict["time"].append(time)
                        result_dict["PDI91201"].append(PDI91201)
                        result_dict["FI91601"].append(FI91601)
                        result_dict["FA"].append(FA)
                        result_dict["L2"].append(L2)
                        result_dict["LIMS_S905_Water"].append(LIMS_S905_Water)
                        result_dict["LIMS_S907_Water"].append(LIMS_S907_Water)
                except:
                    pass

    polynomial_pd = pd.DataFrame(result_dict)

    return polynomial_pd


def MX_linear(FI91601_list, LIMS_S708_Br_Index, LIMS_S708_A9, LIMS_S708_A10, LIMS_S708_Water, LIMS_S708_Sulfur,
              LIMS_s919_A9, LIMS_s919_A10, LIMS_S905_Water, LIMS_S907_Water, FA_list, L2_list, mx):

    slope = [0.5026537152494234, -0.1419981918239352, 0.316846141819883, -174.93106101283536, -30.25005372968611,
             1.6567280752173563e-06, 2.574886695838782, 0.01024542584819767, 0.0014449728612576096,
             0.012527214098019598,
             0.056843854473878845, -171.28765027686828, -112.80014997254257]

    intercept = 203.0815902570037

    result_dict = {
        "method": [],
        "time": [],
        "MX": [],
        "FI91601": [],
        "FA": [],
        "L2": [],
        "LIMS_S905_Water": [],
        "LIMS_S907_Water": [],
        "LIMS_S708_Br_Index":[],
        "LIMS_S708_A9":[],
        "LIMS_S708_A10":[],
        "LIMS_S708_Water":[],
        "LIMS_S708_Sulfur":[],
        "LIMS_s919_A9":[],
        "LIMS_s919_A10":[]
    }

    for FI91601 in FI91601_list:
        for FA in FA_list:
            for L2 in L2_list:

                FI91601_slope = (slope[1] * FI91601)
                LIMS_S708_Br_Index_slope = (slope[2] * LIMS_S708_Br_Index)
                LIMS_S708_A9_slope = (slope[3] * LIMS_S708_A9)
                LIMS_S708_A10_slope = (slope[4] * LIMS_S708_A10)
                LIMS_S708_Water_slope = (slope[5] * LIMS_S708_Water)
                LIMS_S708_Sulfur_slope = (slope[6] * LIMS_S708_Sulfur)
                LIMS_s919_A9_slope = (slope[7] * LIMS_s919_A9)
                LIMS_s919_A10_slope = (slope[8] * LIMS_s919_A10)
                LIMS_S905_Water_slope = (slope[9] * LIMS_S905_Water)
                LIMS_S907_Water_slope = (slope[10] * LIMS_S907_Water)
                FA_slope = (slope[11] * FA)
                L2_slope = (slope[12] * L2)
    
                time = (mx - FI91601_slope - LIMS_S708_Br_Index_slope - LIMS_S708_A9_slope - LIMS_S708_A10_slope - LIMS_S708_Water_slope - LIMS_S708_Sulfur_slope - LIMS_s919_A9_slope - LIMS_s919_A10_slope - LIMS_S905_Water_slope - LIMS_S907_Water_slope - FA_slope - L2_slope - intercept) / slope[0]
                time = round(0.01488613 * time + 14.08157744, 2)

                result_dict["method"].append("Linear")
                result_dict["time"].append(time)
                result_dict["MX"].append(mx)
                result_dict["FI91601"].append(FI91601)
                result_dict["FA"].append(FA)
                result_dict["L2"].append(L2)
                result_dict["LIMS_S905_Water"].append(LIMS_S905_Water)
                result_dict["LIMS_S907_Water"].append(LIMS_S907_Water)
                result_dict["LIMS_S708_Br_Index"].append(LIMS_S708_Br_Index)
                result_dict["LIMS_S708_A9"].append(LIMS_S708_A9)
                result_dict["LIMS_S708_A10"].append(LIMS_S708_A10)
                result_dict["LIMS_S708_Water"].append(LIMS_S708_Water)
                result_dict["LIMS_S708_Sulfur"].append(LIMS_S708_Sulfur)
                result_dict["LIMS_s919_A9"].append(LIMS_s919_A9)
                result_dict["LIMS_s919_A10"].append(LIMS_s919_A10)

    linear_pd = pd.DataFrame(result_dict)

    return linear_pd


def MX_lasso(FI91601_list, LIMS_S708_Br_Index, LIMS_S708_A9, LIMS_S708_A10, LIMS_S708_Water, LIMS_S708_Sulfur,
             LIMS_s919_A9, LIMS_s919_A10, LIMS_S905_Water, LIMS_S907_Water, FA_list, L2_list, mx):

    slope = [0.5329721785420252, -0.1200195308700459, 0.3162080819637333, 0.0, 0.0,
             4.827238991004334e-05, 0.0, 0.01016124930197615, 0.0050229248265834135,
             0.027421442466354105, 0.08645369662933292, 0.0, 0.0]

    intercept = 48.301960786461834

    result_dict = {
        "method": [],
        "time": [],
        "MX": [],
        "FI91601": [],
        "FA": [],
        "L2": [],
        "LIMS_S905_Water": [],
        "LIMS_S907_Water": [],
        "LIMS_S708_Br_Index": [],
        "LIMS_S708_A9": [],
        "LIMS_S708_A10": [],
        "LIMS_S708_Water": [],
        "LIMS_S708_Sulfur": [],
        "LIMS_s919_A9": [],
        "LIMS_s919_A10": []
    }

    for FI91601 in FI91601_list:
        for FA in FA_list:
            for L2 in L2_list:

                FI91601_slope = (slope[1] * FI91601)
                LIMS_S708_Br_Index_slope = (slope[2] * LIMS_S708_Br_Index)
                LIMS_S708_A9_slope = (slope[3] * LIMS_S708_A9)
                LIMS_S708_A10_slope = (slope[4] * LIMS_S708_A10)
                LIMS_S708_Water_slope = (slope[5] * LIMS_S708_Water)
                LIMS_S708_Sulfur_slope = (slope[6] * LIMS_S708_Sulfur)
                LIMS_s919_A9_slope = (slope[7] * LIMS_s919_A9)
                LIMS_s919_A10_slope = (slope[8] * LIMS_s919_A10)
                LIMS_S905_Water_slope = (slope[9] * LIMS_S905_Water)
                LIMS_S907_Water_slope = (slope[10] * LIMS_S907_Water)
                FA_slope = (slope[11] * FA)
                L2_slope = (slope[12] * L2)
    
                time = (mx - FI91601_slope - LIMS_S708_Br_Index_slope - LIMS_S708_A9_slope - LIMS_S708_A10_slope - LIMS_S708_Water_slope - LIMS_S708_Sulfur_slope - LIMS_s919_A9_slope - LIMS_s919_A10_slope - LIMS_S905_Water_slope - LIMS_S907_Water_slope - FA_slope - L2_slope - intercept) / slope[0]
                time = round(0.01658771 * time + 14.05004525, 2)

                result_dict["method"].append("Lasso")
                result_dict["time"].append(time)
                result_dict["MX"].append(mx)
                result_dict["FI91601"].append(FI91601)
                result_dict["FA"].append(FA)
                result_dict["L2"].append(L2)
                result_dict["LIMS_S905_Water"].append(LIMS_S905_Water)
                result_dict["LIMS_S907_Water"].append(LIMS_S907_Water)
                result_dict["LIMS_S708_Br_Index"].append(LIMS_S708_Br_Index)
                result_dict["LIMS_S708_A9"].append(LIMS_S708_A9)
                result_dict["LIMS_S708_A10"].append(LIMS_S708_A10)
                result_dict["LIMS_S708_Water"].append(LIMS_S708_Water)
                result_dict["LIMS_S708_Sulfur"].append(LIMS_S708_Sulfur)
                result_dict["LIMS_s919_A9"].append(LIMS_s919_A9)
                result_dict["LIMS_s919_A10"].append(LIMS_s919_A10)

    lasso_pd = pd.DataFrame(result_dict)

    return lasso_pd


def MX_polynomial(FI91601_list, LIMS_S708_Br_Index, LIMS_S708_A9, LIMS_S708_A10, LIMS_S708_Water, LIMS_S708_Sulfur,
                  LIMS_s919_A9, LIMS_s919_A10, LIMS_S905_Water, LIMS_S907_Water, FA_list, L2_list, mx):

    slope = [0.5026537152494067, -0.14199819182393636, 0.31684614181929305, -174.9310610125306,
             -30.250053729715844, 1.6567280624497915e-06, 2.5748866958402066, 0.010245425848243665,
             0.0014449728612592194, 0.012527214097986341, 0.05684385447392647, -171.28765027686774, -112.80014997254285]

    intercept = 203.08159025698615

    result_dict = {
        "method": [],
        "time": [],
        "MX": [],
        "FI91601": [],
        "FA": [],
        "L2": [],
        "LIMS_S905_Water": [],
        "LIMS_S907_Water": [],
        "LIMS_S708_Br_Index": [],
        "LIMS_S708_A9": [],
        "LIMS_S708_A10": [],
        "LIMS_S708_Water": [],
        "LIMS_S708_Sulfur": [],
        "LIMS_s919_A9": [],
        "LIMS_s919_A10": []
    }

    for FI91601 in FI91601_list:
        for FA in FA_list:
            for L2 in L2_list:
                FI91601_slope = (slope[1] * FI91601)
                LIMS_S708_Br_Index_slope = (slope[2] * LIMS_S708_Br_Index)
                LIMS_S708_A9_slope = (slope[3] * LIMS_S708_A9)
                LIMS_S708_A10_slope = (slope[4] * LIMS_S708_A10)
                LIMS_S708_Water_slope = (slope[5] * LIMS_S708_Water)
                LIMS_S708_Sulfur_slope = (slope[6] * LIMS_S708_Sulfur)
                LIMS_s919_A9_slope = (slope[7] * LIMS_s919_A9)
                LIMS_s919_A10_slope = (slope[8] * LIMS_s919_A10)
                LIMS_S905_Water_slope = (slope[9] * LIMS_S905_Water)
                LIMS_S907_Water_slope = (slope[10] * LIMS_S907_Water)
                FA_slope = (slope[11] * FA)
                L2_slope = (slope[12] * L2)

                time = (mx - FI91601_slope - LIMS_S708_Br_Index_slope - LIMS_S708_A9_slope - LIMS_S708_A10_slope - LIMS_S708_Water_slope - LIMS_S708_Sulfur_slope - LIMS_s919_A9_slope - LIMS_s919_A10_slope - LIMS_S905_Water_slope - LIMS_S907_Water_slope - FA_slope - L2_slope - intercept) / slope[0]
                time = round(0.01488613 * time + 14.08157744, 2)

                result_dict["method"].append("Polynomial")
                result_dict["time"].append(time)
                result_dict["MX"].append(mx)
                result_dict["FI91601"].append(FI91601)
                result_dict["FA"].append(FA)
                result_dict["L2"].append(L2)
                result_dict["LIMS_S905_Water"].append(LIMS_S905_Water)
                result_dict["LIMS_S907_Water"].append(LIMS_S907_Water)
                result_dict["LIMS_S708_Br_Index"].append(LIMS_S708_Br_Index)
                result_dict["LIMS_S708_A9"].append(LIMS_S708_A9)
                result_dict["LIMS_S708_A10"].append(LIMS_S708_A10)
                result_dict["LIMS_S708_Water"].append(LIMS_S708_Water)
                result_dict["LIMS_S708_Sulfur"].append(LIMS_S708_Sulfur)
                result_dict["LIMS_s919_A9"].append(LIMS_s919_A9)
                result_dict["LIMS_s919_A10"].append(LIMS_s919_A10)

    polynomial_pd = pd.DataFrame(result_dict)

    return polynomial_pd

