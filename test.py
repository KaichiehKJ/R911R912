
import pandas as pd


df_1 = pd.read_excel("data/data_1.xlsx", sheet_name = "工作表1")
df_2 = pd.read_excel("data/data_2.xlsx", sheet_name = "工作表1")

for item in df_2["time"].to_list():
    index = df_1[df_1["time"] == item].index
    # print(df_1.loc[index-7,df_1.columns[1:]])
    # print(df_2.loc[df_2["time"] == item, df_2.columns[1:]].values.tolist())
    df_1.loc[index-7,df_1.columns[1:]] = df_2.loc[df_2["time"] == item, df_2.columns[1:]].values.tolist()

df_1.to_excel("data/data_3.xlsx")
