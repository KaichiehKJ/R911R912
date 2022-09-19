import re
from django.shortcuts import render
from .calculator import (
    R911_linear, R911_lasso, R911_polynomial,
    R912_linear, R912_lasso, R912_polynomial,
    MX_linear, MX_lasso, MX_polynomial
    )

# Create your views here.

def r911(request):
    
    context = {
        "Linear":0,
        "Lasso":0,
        "Polynomial":0
    }

    if request.method == "POST":

        FI91601 = float(request.POST.get("FI91601", ""))
        LIMS_S905_Water = float(request.POST.get("LIMS-S905@Water", ""))
        LIMS_S907_Water = float(request.POST.get("LIMS-S907@Water", ""))
        FA = float(request.POST.get("FA", ""))
        L2 = float(request.POST.get("L2", ""))
        PDI91101 = float(request.POST.get("PDI91101", ""))

        context["Linear"] = R911_linear(FI91601, LIMS_S905_Water, LIMS_S907_Water, FA, L2, PDI91101)
        context["Lasso"] = R911_lasso(FI91601, LIMS_S905_Water, LIMS_S907_Water, FA, L2, PDI91101)
        context["Polynomial"] = R911_polynomial(FI91601, LIMS_S905_Water, LIMS_S907_Water, FA, L2, PDI91101)

        return render(request, "r911.html", context)

    return render(request, "r911.html", context)

def r912(request):
    
    context = {
        "Linear":0,
        "Lasso":0,
        "Polynomial":0
    }

    if request.method == "POST":

        FI91601 = float(request.POST.get("FI91601", ""))
        LIMS_S905_Water = float(request.POST.get("LIMS-S905@Water", ""))
        LIMS_S907_Water = float(request.POST.get("LIMS-S907@Water", ""))
        FA = float(request.POST.get("FA", ""))
        L2 = float(request.POST.get("L2", ""))
        PDI91201 = float(request.POST.get("PDI91201", ""))

        context["Linear"] = R912_linear(FI91601, LIMS_S905_Water, LIMS_S907_Water, FA, L2, PDI91201)
        context["Lasso"] = R912_lasso(FI91601, LIMS_S905_Water, LIMS_S907_Water, FA, L2, PDI91201)
        context["Polynomial"] = R912_polynomial(FI91601, LIMS_S905_Water, LIMS_S907_Water, FA, L2, PDI91201)

        return render(request, "r912.html", context)

    return render(request, "r912.html", context)

def mx(request):

    context = {
        "Linear":0,
        "Lasso":0,
        "Polynomial":0
    }

    if request.method == "POST":

        FI91601 = float(request.POST.get("FI91601", ""))
        LIMS_S708_Br_Index = float(request.POST.get("LIMS-S708@Br.Index", ""))
        LIMS_S708_A9 = float(request.POST.get("LIMS-S708@A9", ""))
        LIMS_S708_A10 = float(request.POST.get("LIMS-S708@A10+", ""))
        LIMS_S708_Water = float(request.POST.get("LIMS-S708@Water", ""))
        LIMS_S708_Sulfur = float(request.POST.get("LIMS-S708@Sulfur", ""))
        LIMS_s919_A9 = float(request.POST.get("LIMS-s919@A9", ""))
        LIMS_s919_A10 = float(request.POST.get("LIMS-s919@A10+", ""))
        LIMS_S905_Water = float(request.POST.get("LIMS-S905@Water", ""))
        LIMS_S907_Water = float(request.POST.get("LIMS-S907@Water", ""))
        FA = float(request.POST.get("FA", ""))
        L2 = float(request.POST.get("L2", ""))
        mx = float(request.POST.get("mx", ""))

        context["Linear"] = MX_linear(FI91601, LIMS_S708_Br_Index, LIMS_S708_A9, LIMS_S708_A10, LIMS_S708_Water, LIMS_S708_Sulfur, 
                                      LIMS_s919_A9, LIMS_s919_A10, LIMS_S905_Water, LIMS_S907_Water, FA, L2, mx)
        context["Lasso"] = MX_lasso(FI91601, LIMS_S708_Br_Index, LIMS_S708_A9, LIMS_S708_A10, LIMS_S708_Water, LIMS_S708_Sulfur, 
                                    LIMS_s919_A9, LIMS_s919_A10, LIMS_S905_Water, LIMS_S907_Water, FA, L2, mx)
        context["Polynomial"] = MX_polynomial(FI91601, LIMS_S708_Br_Index, LIMS_S708_A9, LIMS_S708_A10, LIMS_S708_Water, LIMS_S708_Sulfur, 
                                              LIMS_s919_A9, LIMS_s919_A10, LIMS_S905_Water, LIMS_S907_Water, FA, L2, mx)

        return render(request, "mx.html", context)

    return render(request, "mx.html", context)