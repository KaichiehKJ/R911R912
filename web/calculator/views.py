import re
from django.shortcuts import render
from .calculator import (
    R911_linear, R911_lasso, R911_polynomial,
    R912_linear, R912_lasso, R912_polynomial,
    MX_linear, MX_lasso, MX_polynomial
    )

import numpy as np
import pandas as pd

# Create your views here.

def home(request):
    return render(request, "home.html")


def r911(request):
    
    context = {
        "analysis":"(unanalysis)"
    }

    if request.method == "POST":

        FI91601_max = float(request.POST.get("FI91601_max", ""))
        FI91601_min = float(request.POST.get("FI91601_min", ""))
        FI91601_list = [FI91601_max, FI91601_min]

        LIMS_S905_Water = float(request.POST.get("LIMS-S905@Water", ""))
        LIMS_S907_Water = float(request.POST.get("LIMS-S907@Water", ""))

        FA_max = float(request.POST.get("FA_max", ""))
        FA_min = float(request.POST.get("FA_min", ""))
        FA_step = float(request.POST.get("FA_step", ""))
        FA_list = np.arange(FA_min, FA_max, FA_step)
        if FA_max not in FA_list:
            FA_list = np.append(FA_list, FA_max)


        L2_max = float(request.POST.get("L2_max", ""))
        L2_min = float(request.POST.get("L2_min", ""))
        L2_step = float(request.POST.get("L2_step", ""))
        L2_list = np.arange(L2_min, L2_max, L2_step)
        if L2_max not in L2_list:
            L2_list = np.append(L2_list, L2_max)

        PDI91101 = float(request.POST.get("PDI91101", ""))

        linear_pd = R911_linear(FI91601_list = FI91601_list, LIMS_S905_Water = LIMS_S905_Water, LIMS_S907_Water = LIMS_S907_Water,
                                FA_list = FA_list, L2_list = L2_list, PDI91101 = PDI91101)
        lasso_pd = R911_lasso(FI91601_list = FI91601_list, LIMS_S905_Water = LIMS_S905_Water, LIMS_S907_Water = LIMS_S907_Water,
                              FA_list = FA_list, L2_list = L2_list, PDI91101 = PDI91101)
        polynomial_pd = R911_polynomial(FI91601_list = FI91601_list, LIMS_S905_Water = LIMS_S905_Water, LIMS_S907_Water = LIMS_S907_Water,
                                        FA_list = FA_list, L2_list = L2_list, PDI91101 = PDI91101)

        new_df = pd.DataFrame()
        for df in [linear_pd, lasso_pd, polynomial_pd]:
            new_df = pd.concat([new_df, df])

        new_df.to_csv("web/static/result/r911_time.csv", index=False)

        context["analysis"] = "(finish)"
        return render(request, "r911.html", context)

    return render(request, "r911.html", context)

def r912(request):

    context = {
        "analysis": "(unanalysis)"
    }

    if request.method == "POST":

        FI91601_max = float(request.POST.get("FI91601_max", ""))
        FI91601_min = float(request.POST.get("FI91601_min", ""))
        FI91601_list = [FI91601_max, FI91601_min]

        LIMS_S905_Water = float(request.POST.get("LIMS-S905@Water", ""))
        LIMS_S907_Water = float(request.POST.get("LIMS-S907@Water", ""))

        FA_max = float(request.POST.get("FA_max", ""))
        FA_min = float(request.POST.get("FA_min", ""))
        FA_step = float(request.POST.get("FA_step", ""))
        FA_list = np.arange(FA_min, FA_max, FA_step)
        if FA_max not in FA_list:
            FA_list = np.append(FA_list, FA_max)

        L2_max = float(request.POST.get("L2_max", ""))
        L2_min = float(request.POST.get("L2_min", ""))
        L2_step = float(request.POST.get("L2_step", ""))
        L2_list = np.arange(L2_min, L2_max, L2_step)
        if L2_max not in L2_list:
            L2_list = np.append(L2_list, L2_max)

        PDI91201 = float(request.POST.get("PDI91201", ""))

        linear_pd = R912_linear(FI91601_list=FI91601_list, LIMS_S905_Water=LIMS_S905_Water,
                                LIMS_S907_Water=LIMS_S907_Water,
                                FA_list=FA_list, L2_list=L2_list, PDI91201=PDI91201)
        lasso_pd = R912_lasso(FI91601_list=FI91601_list, LIMS_S905_Water=LIMS_S905_Water,
                              LIMS_S907_Water=LIMS_S907_Water,
                              FA_list=FA_list, L2_list=L2_list, PDI91201=PDI91201)
        polynomial_pd = R912_polynomial(FI91601_list=FI91601_list, LIMS_S905_Water=LIMS_S905_Water, LIMS_S907_Water=LIMS_S907_Water,
                                        FA_list=FA_list, L2_list=L2_list, PDI91201=PDI91201)

        new_df = pd.DataFrame()
        for df in [linear_pd, lasso_pd, polynomial_pd]:
            new_df = pd.concat([new_df, df])

        new_df.to_csv("web/static/result/r912_time.csv", index=False)

        context["analysis"] = "(finish)"
        return render(request, "r912.html", context)

    return render(request, "r912.html", context)

def mx(request):

    context = {
        "analysis": "(unanalysis)"
    }

    if request.method == "POST":

        FI91601_max = float(request.POST.get("FI91601_max", ""))
        FI91601_min = float(request.POST.get("FI91601_min", ""))
        FI91601_list = [FI91601_max, FI91601_min]

        LIMS_S708_Br_Index = float(request.POST.get("LIMS-S708@Br.Index", ""))
        LIMS_S708_A9 = float(request.POST.get("LIMS-S708@A9", ""))
        LIMS_S708_A10 = float(request.POST.get("LIMS-S708@A10+", ""))
        LIMS_S708_Water = float(request.POST.get("LIMS-S708@Water", ""))
        LIMS_S708_Sulfur = float(request.POST.get("LIMS-S708@Sulfur", ""))
        LIMS_s919_A9 = float(request.POST.get("LIMS-s919@A9", ""))
        LIMS_s919_A10 = float(request.POST.get("LIMS-s919@A10+", ""))
        LIMS_S905_Water = float(request.POST.get("LIMS-S905@Water", ""))
        LIMS_S907_Water = float(request.POST.get("LIMS-S907@Water", ""))

        FA_max = float(request.POST.get("FA_max", ""))
        FA_min = float(request.POST.get("FA_min", ""))
        FA_step = float(request.POST.get("FA_step", ""))
        FA_list = np.arange(FA_min, FA_max, FA_step)
        if FA_max not in FA_list:
            FA_list = np.append(FA_list, FA_max)

        L2_max = float(request.POST.get("L2_max", ""))
        L2_min = float(request.POST.get("L2_min", ""))
        L2_step = float(request.POST.get("L2_step", ""))
        L2_list = np.arange(L2_min, L2_max, L2_step)
        if L2_max not in L2_list:
            L2_list = np.append(L2_list, L2_max)


        mx = float(request.POST.get("MX", ""))
        
        linear_pd = MX_linear(FI91601_list, LIMS_S708_Br_Index, LIMS_S708_A9, LIMS_S708_A10, LIMS_S708_Water, LIMS_S708_Sulfur,
                              LIMS_s919_A9, LIMS_s919_A10, LIMS_S905_Water, LIMS_S907_Water, FA_list, L2_list, mx)
        lasso_pd = MX_lasso(FI91601_list, LIMS_S708_Br_Index, LIMS_S708_A9, LIMS_S708_A10, LIMS_S708_Water, LIMS_S708_Sulfur,
                            LIMS_s919_A9, LIMS_s919_A10, LIMS_S905_Water, LIMS_S907_Water, FA_list, L2_list, mx)
        polynomial_pd = MX_polynomial(FI91601_list, LIMS_S708_Br_Index, LIMS_S708_A9, LIMS_S708_A10, LIMS_S708_Water, LIMS_S708_Sulfur,
                                      LIMS_s919_A9, LIMS_s919_A10, LIMS_S905_Water, LIMS_S907_Water, FA_list, L2_list, mx)

        new_df = pd.DataFrame()
        for df in [linear_pd, lasso_pd, polynomial_pd]:
            new_df = pd.concat([new_df, df])

        new_df.to_csv("web/static/result/mx_time.csv", index=False)

        context["analysis"] = "(finish)"
        return render(request, "mx.html", context)

    return render(request, "mx.html", context)