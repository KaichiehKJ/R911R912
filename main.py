
import joblib
from sklearn.metrics import r2_score
import torch
from torch.utils.data import TensorDataset,DataLoader
import numpy as np
import pandas as pd
from model.reinforcement import R2D2, ActorCritic
from model.CNNLSTM import cnn_lstm
import matplotlib.pyplot as plt

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
                                                                  
                                                                  
    a = pd.DataFrame(self.test_data['value']).describe()
    self.test_data['request'] = None
  
    for i in a.columns:
      if self.test_data['request'] == None:
         self.test_data['request'] = np.random.uniform(low = a.loc['50%'][i],
                                                        high=a.loc['max'][i],
                                                        size=(self.test_data['value'].shape[0],1))
      else:
        self.test_data['request'] = np.hstack([self.test_data['request'],
                                                np.random.uniform(low = a.loc['50%'][i],
                                                                  high = a.loc['max'][i],
                                                                  size = (self.test_data['value'].shape[0],1))])
    
  def np_convert_tensor(self, dataset):
    # tensor
    for item in ["state", "request", "action", "value"]:
      dataset[item] = torch.FloatTensor(dataset[item])

    return dataset
    
    
  def set_iter(self, dataset):
    
    tensor_Dataset = TensorDataset(dataset['state'], dataset['request'], dataset['action'], dataset['value'])
    data_iter = DataLoader(tensor_Dataset, batch_size = self.batch_size, shuffle=True)
    
    return data_iter
    
  def creat_iter(self):
    
    self.set_data_request()
    
    self.train_data = self.np_convert_tensor(dataset = self.train_data)
    self.test_data = self.np_convert_tensor(dataset = self.test_data)
    
    self.train_iter = self.set_iter(dataset = self.train_data)
    self.test_iter = self.set_iter(dataset = self.test_data)
    
  def init_weights(self, m):
    if hasattr(m,'weight'):
        try:
            torch.nn.init.xavier_uniform(m.weight)
        except:
            pass
    
    if hasattr(m,'bias'):
        try:
            m.bias.data.fill_(0.1)
        except:
            pass
  
  
  def set_model(self, device, method, hidden_size, time_step, bidirectional):
    
    if method == "reinforcement":
      actor = R2D2(self.state_size + self.request_size, self.action_size, self.hidden_size)
      critic = R2D2(self.state_size + self.action_size, self.value_size, self.hidden_size)
      self.model = ActorCritic(actor, critic, self.time_step, device, self.gpu_id)
      self.model.cuda(device = self.gpu_id)
    elif method == "cnn_lstm":
      self.model = cnn_lstm(device = device, hidden_size = hidden_size, time_step = time_step, bidirectional = bidirectional)
      self.model.cuda(device = self.gpu_id)
      self.model.apply(self.init_weights)
      
  
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
        
  def train_lstm(self, epochs):
    
    loss_fn = torch.nn.L1Loss() #
    optimiser = torch.optim.Adam(self.model.parameters(), lr=0.001)
    train_history = []
    
    for epoch in range(epochs):
      epoch_loss = 0
      for i, (bs, _, ba, bv) in enumerate(self.train_iter):
        if i <= 17: #206, (445, 195), (470) 178 256
          y_pred = self.model(state = bs.cuda(device = self.gpu_id), action = ba.cuda(device = self.gpu_id))
          loss = loss_fn(y_pred, bv.cuda(device = self.gpu_id))
          # update weights
          optimiser.zero_grad()
          loss.backward()
          optimiser.step()
          
          epoch_loss += loss.item()
      
      train_history.append(epoch_loss/17)
      if epoch % 10 == 0:
        print('epoch:{} loss:{}'.format(epoch, epoch_loss/17))
        
  def pred_test(self):
    
    def MAE(true, pred):
      return np.mean(np.abs(true-pred))
    
    def mape(a, b): 
      mask = a != 0
      return (np.fabs(a - b)/a)[mask].mean()
    
    def rmse(true, pred):
      return np.sqrt(((pred - true) ** 2).mean())
    
    real_value = []
    pred_value = []
    self.model.eval()
    
    for i in range(66):
      s,r,a,v = next(iter(self.test_iter))
      y_pred = self.model(state = s.cuda(device = self.gpu_id), action = a.cuda(device = self.gpu_id))
      
      pred_value.append(y_pred.cpu().detach().numpy()[0])
      real_value.append(v.detach().numpy()[0])
    
    pred_value = np.array(pred_value)
    real_value = np.array(real_value)
    mm_y = self.dataset['mm_scale']["mm_value"]
    pred_value = mm_y.inverse_transform(pred_value.reshape(-1, 1)).reshape(-1,1)
    real_value = mm_y.inverse_transform(real_value.reshape(-1, 1)).reshape(-1,1)
    
    print('r2', r2_score(real_value[:,0], pred_value[:,0]))
    print('MAPE',mape(real_value[:,0],pred_value[:,0]))
    print("RMSE:", rmse(real_value[:,0], pred_value[:,0]))
    
    torch.save(self.model.state_dict(),'result/model/R912.pth')
    
    plt.figure(figsize=(20,5))
    plt.plot(pd.Series(pred_value[:,0]).rolling(1).mean(),label='pred')
    plt.plot(pd.Series(real_value[:,0]).rolling(1).mean(),label='real')
    plt.legend()
    plt.savefig('result/plot/plot.png')


if __name__=="__main__":
  
  path = "result/pre_process_data/ARO2-LIMS-s922@MX_dataset.pkl"
  time_step = 4
  batch_size = 32
  hidden_size = 128
  epochs = 100
  gpu_id = 6
  device = "cuda:6"
  bidirectional = False

  main = Main(path = path, batch_size = batch_size, hidden_size = hidden_size, gpu_id = gpu_id)
  main.creat_iter()
  main.set_model(device = device, method = "cnn_lstm", hidden_size = hidden_size, time_step = time_step, bidirectional = bidirectional)
  main.train_lstm(epochs = epochs)
  main.pred_test()


