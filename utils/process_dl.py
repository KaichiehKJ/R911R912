
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib

class process():
  
  def __init__(self, path, a_col, s_col, y_col, time_step, scale_method):
    
    self.y_status = False
    
    self.a_col = a_col
    self.s_col = s_col
    self.y_col = y_col
    self.time_step = time_step
    self.load_xlsx(path = path)
    self.set_scaler(scale_method = scale_method)
    
  def set_dataset_dict(self):
    
    self.train_data = {key:[] for key in ["state", "action", "value"]}
    self.test_data = {key:[] for key in ["state", "action", "value"]}
  
  def load_xlsx(self, path):
    
    self.df_x = pd.read_excel(path, sheet_name = "R911R912_TAG相關錶點資料(X)")
    
    if y_col[0] not in self.df_x.columns.to_list():
      self.df_y = pd.read_excel(path, sheet_name= "目標值(Y)相關錶點資料")
      self.y_status = True
  
  def set_scaler(self, scale_method):
    
    if scale_method == "MinMax":
      self.mm_a = MinMaxScaler()
      self.mm_s = MinMaxScaler()
      self.mm_y = MinMaxScaler()
    elif scale_method == "Standard":
      self.mm_a = StandardScaler()
      self.mm_s = StandardScaler()
      self.mm_y = StandardScaler()
    
  def select_col(self):
    
    if self.y_status is False:
      self.df_y = self.df_x[y_col]
    else:
      self.df_y = self.df_y[y_col]
    
    self.df_x = self.df_x[self.s_col + self.a_col]
    
  def cover_numeeric(self, df):
    
    for col in df.columns:
      df[col] = pd.to_numeric(df[col], errors='coerce')
      
    return df
      
  def fill_data(self, method):
    
    if method == "fill":
      self.df_x = self.df_x.fillna(method='ffill')
      self.df_x = self.df_x.fillna(method='bfill')
    else:
      pass
    
  def merge_data(self):
    
    self.df_xy = self.df_x.join(self.df_y)

  def exclude_value(self, x_conditon, y_condition):
    
    self.df_xy = self.df_xy.loc[(self.df_xy["ARO2-DCS-FI91601"] > x_conditon),:]
    if self.y_col[0] == "ARO2-LIMS-s922@MX":
      self.df_xy.loc[self.df_xy[self.y_col[0]] > y_condition, self.y_col[0]] = np.nan
    else:
      self.df_xy = self.df_xy[self.df_xy.loc[:, self.y_col[0]] != "I/O Timeout"] 
      self.df_xy = self.df_xy.loc[self.df_xy[self.y_col[0]] < y_condition, :]
      
    self.df_xy.reset_index(drop = True, inplace = True)
    self.df_xy = self.cover_numeeric(df = self.df_xy)
    
  def pre_process_df(self, method, x_conditon, y_condition):
    
    self.select_col()
    self.df_x = self.cover_numeeric(df = self.df_x)
    self.fill_data(method = method)
    self.merge_data()
    self.exclude_value(x_conditon = x_conditon, y_condition = y_condition)

  def set_selectindex_cutpoint(self, proportion = 0.95):
    
    self.select_index = self.df_xy.loc[self.df_xy[self.y_col[0]].notnull(), self.y_col[0]].index.values.tolist()
    self.cut_point = int(len(self.select_index) * proportion)
  
  def scale_transform(self):
    
    # for i in range(0,8): # T 4:(3+1) 8:(7+1)
    #   if i == 0:
    #     train_index = self.select_index[:self.cut_point]
    #     test_index = self.select_index[self.cut_point:]
    #   else:
    #     train_index = train_index + (np.array(self.select_index[:self.cut_point]) -i).tolist()
    #     test_index = test_index + (np.array(self.select_index[self.cut_point:]) -i).tolist()
    # 
    # train_index = sorted(train_index)
    # test_index = sorted(test_index)
    # 
    # self.mm_a.fit(self.df_xy.loc[train_index, self.a_col])
    # self.mm_s.fit(self.df_xy.loc[train_index, self.s_col])
    # self.mm_y.fit(self.df_xy.loc[train_index, self.y_col])
    # 
    # # train
    # self.df_xy.loc[train_index, self.a_col] = self.mm_a.transform(self.df_xy.loc[train_index, self.a_col])
    # self.df_xy.loc[train_index, self.s_col] = self.mm_s.transform(self.df_xy.loc[train_index, self.s_col])
    # self.df_xy.loc[train_index, self.y_col[0]] = self.mm_y.transform(self.df_xy.loc[train_index, self.y_col])[:,0]
    # # test
    # self.df_xy.loc[test_index, self.a_col] = self.mm_a.transform(self.df_xy.loc[test_index, self.a_col])
    # self.df_xy.loc[test_index, self.s_col] = self.mm_s.transform(self.df_xy.loc[test_index, self.s_col])
    # self.df_xy.loc[test_index, self.y_col[0]] = self.mm_y.transform(self.df_xy.loc[test_index, self.y_col])[:,0]
    
    self.mm_a.fit(self.df_xy.loc[:self.select_index[self.cut_point], self.a_col])
    self.mm_s.fit(self.df_xy.loc[:self.select_index[self.cut_point], self.s_col])
    self.mm_y.fit(self.df_xy.loc[:self.select_index[self.cut_point], self.y_col])

    # train
    self.df_xy.loc[:self.select_index[self.cut_point], self.a_col] = self.mm_a.transform(self.df_xy.loc[:self.select_index[self.cut_point], self.a_col])
    self.df_xy.loc[:self.select_index[self.cut_point], self.s_col] = self.mm_s.transform(self.df_xy.loc[:self.select_index[self.cut_point], self.s_col])
    self.df_xy.loc[:self.select_index[self.cut_point], self.y_col[0]] = self.mm_y.transform(self.df_xy.loc[:self.select_index[self.cut_point], self.y_col])[:,0]
    # test
    self.df_xy.loc[self.select_index[self.cut_point]:, self.a_col] = self.mm_a.transform(self.df_xy.loc[self.select_index[self.cut_point]:, self.a_col])
    self.df_xy.loc[self.select_index[self.cut_point]:, self.s_col] = self.mm_s.transform(self.df_xy.loc[self.select_index[self.cut_point]:, self.s_col])
    self.df_xy.loc[self.select_index[self.cut_point]:, self.y_col[0]] = self.mm_y.transform(self.df_xy.loc[self.select_index[self.cut_point]:, self.y_col])[:,0]
  
  def dataset_index(self):
    
    # self.train_index = [i for i in self.select_index[:self.cut_point]]
    # self.test_index = [i for i in self.select_index[self.cut_point:]]
    self.train_index = [i for i in range(self.time_step-1, self.cut_point, self.time_step)]
    self.test_index = [i for i in range(self.cut_point, len(self.select_index), self.time_step)]
  
  def split_dataset(self, index_list):
    
    # if data_sets == "train":
    #   index_list = self.select_index[:self.cut_point]
    # else:
    #   index_list = self.select_index[self.cut_point:]
    
    dataset_dict = {key:[] for key in ["state", "action", "value"]}
    for i in index_list:
      dataset_dict["state"].append(self.df_xy.loc[(i-(self.time_step-1)):i, self.s_col])
      dataset_dict["action"].append(self.df_xy.loc[(i-(self.time_step-1)):i, self.a_col])
      dataset_dict["value"].append(self.df_xy.loc[i, self.y_col])
    
    for key, value in  dataset_dict.items():
      dataset_dict[key] = np.array(value)
    
    return dataset_dict
  
  def store_mm_scaler(self):
    
    mm_scale_dict = {"mm_state":self.mm_s, "mm_action":self.mm_a, "mm_value":self.mm_y}
    
    return mm_scale_dict
    
  def store_col_dict(self):
    
    col_dict = {"state_col":self.s_col, "action_col":self.a_col, "value_col":self.y_col}
    
    return col_dict

  def set_dataset(self, proportion):
    
    self.set_selectindex_cutpoint(proportion = proportion)
    self.dataset_index()
    self.scale_transform()
    
    self.train_data = self.split_dataset(index_list = self.train_index)
    self.test_data = self.split_dataset(index_list = self.test_index)

  def save_to_pkl(self):
    
    data = {
      "train_data":self.train_data,
      "test_data":self.test_data,
      "mm_scale":self.store_mm_scaler(),
      "col_name":self.store_col_dict()
      }
      
    path = "result/pre_process_data/{target_name}_dataset.pkl".format(target_name = self.y_col[0])
    joblib.dump(data, path)
  


if __name__=="__main__":
  
  a_col = [
    "ARO2-DCS-R911_2_A_FA", "ARO2-DCS-R911_2_HF", "ARO2-DCS-R911_2_L2_A",
    "ARO2-DCS-R911_2_L3_A", "ARO2-DCS-R911_2_L4_A", "ARO2-DCS-R911_2_XF"
    ]


  s_col  = [
    "ARO2-DCS-FI914A2", "ARO2-DCS-FI914A3", "ARO2-DCS-FI91601",
    "ARO2-DCS-FI91701", "ARO2-DCS-FI93201", "ARO2-DCS-FI94701",
    "ARO2-LIMS-S708@Br.Index", "ARO2-LIMS-S708@A9", "ARO2-LIMS-S708@A10+",
    "ARO2-LIMS-S708@Water", "ARO2-LIMS-S708@Sulfur", "ARO2-LIMS-S905@Water",
    "ARO2-LIMS-S907@Water"
    ]
  
  y_col = ["ARO2-LIMS-s922@MX"]  # "ARO2-LIMS-s922@MX" "ARO2-DCS-PDI91101" "ARO2-DCS-PDI91201"
  
  scale_method = "Standard"
  method = "fill"
  x_conditon = 470
  y_condition = 4 # 1700 4
  proportion = 0.9
  time_step = 4
  
  Process = process(path = "data/R911R912 _明志蔡教授_R4-ARO2.xlsx", a_col = a_col, s_col = s_col, y_col = y_col, time_step = time_step, scale_method = scale_method)
  Process.pre_process_df(method = method, x_conditon = x_conditon, y_condition = y_condition)
  Process.set_dataset(proportion = proportion)
  Process.save_to_pkl()




