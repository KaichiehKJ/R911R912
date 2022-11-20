

import torch
from torch.utils.data import TensorDataset,DataLoader
import joblib
import numpy as np
import pandas as pd
# from tqdm import tqdm_notebook as tqdm
from model.reinforcement import R2D2, ActorCritic


class Main():
  def __init__(self, path, batch_size, hidden_size, gpu_id):
    
    self.load_pkl(path)
    self.set_datasets()
    self.set_parameter(batch_size, hidden_size)
    self.gpu_id = gpu_id
    
  
  def load_pkl(self, path):
    
    self.dataset = joblib.load(path)
    
  def set_datasets(self):
    
    self.train_data = self.dataset["train_data"]
    self.test_data = self.dataset["test_data"]
    
  def set_parameter(self, batch_size, hidden_size):
    
    self.batch_size = batch_size
    self.hidden_size = hidden_size
    self.state_size = self.train_data['state'].shape[2]
    self.request_size = self.train_data['value'].shape[1]
    self.value_size = self.train_data['value'].shape[1]
    self.action_size = self.train_data['action'].shape[2]
    self.time_step = self.train_data['state'].shape[1]
    
  def set_data_request(self):
    
    a = pd.DataFrame(self.train_data['value']).describe()
    self.train_data['request'] = None
  
    for i in a.columns:
      if self.train_data['request'] == None:
         self.train_data['request'] = np.random.uniform(low = a.loc['50%'][i],
                                                        high=a.loc['max'][i],
                                                        size=(self.train_data['value'].shape[0],1))
      else:
        self.train_data['request'] = np.hstack([self.train_data['request'],
                                                np.random.uniform(low = a.loc['50%'][i],
                                                                  high = a.loc['max'][i],
                                                                  size = (self.train_data['value'].shape[0],1))])
    
  def np_convert_tensor(self):
    # tensor
    for item in ["state", "request", "action", "value"]:
      self.train_data[item] = torch.FloatTensor(self.train_data[item])
    
  def set_iter(self):
    
    train_data = TensorDataset(self.train_data['state'], self.train_data['request'], self.train_data['action'], self.train_data['value'])
    self.train_iter = DataLoader(train_data, batch_size = self.batch_size, shuffle=True)
    
  def creat_iter(self):
    
    self.set_data_request()
    self.np_convert_tensor()
    self.set_iter()
    
    
  def set_model(self, device):
    
    actor = R2D2(self.state_size + self.request_size, self.action_size, self.hidden_size)
    critic = R2D2(self.state_size + self.action_size, self.value_size, self.hidden_size)
    self.model = ActorCritic(actor, critic, self.time_step, device, self.gpu_id)
    self.model.cuda(device = self.gpu_id)
    
  
  def train(self, epochs):
    train_history = {}
    train_history['actor'] = []
    train_history['critic'] = []
  
    for epoch in range(epochs):
      critic_loss = 0
      for i, (bs, br, ba, bv) in enumerate(self.train_iter):
        critic_loss += self.model.critic_loss_fun(state = bs.cuda(device = self.gpu_id), action = ba.cuda(device = self.gpu_id), value = bv)

      train_history['critic'].append(critic_loss)
      if epoch % 10 == 0:
        print('epoch:{} critic_loss:{}'.format(epoch, critic_loss))
        
    
    for epoch in range(epochs):
      actor_loss = 0
      for i,(bs,br,ba,bv) in enumerate(self.train_iter):
        actor_loss += self.model.actor_loss_fun(state = bs.cuda(device = self.gpu_id), request = br.cuda(device = self.gpu_id), action_size = self.action_size)
      
      train_history['actor'].append(actor_loss)
      if epoch % 10 == 0:
        print('epoch:{} actor_loss:{}'.format(epoch, actor_loss))



if __name__=="__main__":
  
  path = "result/pre_process_data/ARO2-LIMS-s922@MX_dataset.pkl"
  
  batch_size = 32
  hidden_size = 128
  epochs = 100
  gpu_id = 6
  device = "cuda:6"
  
  main = Main(path = path, batch_size = batch_size, hidden_size = hidden_size, gpu_id = gpu_id)
  main.creat_iter()
  main.set_model(device = device)
  main.train(epochs = epochs)



