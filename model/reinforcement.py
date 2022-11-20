
import torch
import torch.nn as nn
# import torchsnooper
import requests
exec(requests.get('https://raw.githubusercontent.com/facebookresearch/madgrad/main/madgrad/madgrad.py').text)

# @torchsnooper.snoop()
class R2D2(nn.Module):
  def __init__(self, obs_size, n_actions, hidden_size):
    super().__init__()
    self.hidden_size = hidden_size
    self.l1 = nn.Linear(obs_size,self.hidden_size)
    self.l2 = nn.LSTMCell(self.hidden_size,self.hidden_size)
    self.l3 = nn.Linear(self.hidden_size,n_actions)
  
  def forward(self,x,hx,cx):
    x = torch.tanh(self.l1(x))
    hx,cx = self.l2(x,(hx,cx))
    x = torch.tanh(hx)
    x = torch.sigmoid(self.l3(x)) # to range[0,1]
    return x,hx,cx
  
# @torchsnooper.snoop()
class ActorCritic(nn.Module):
  def __init__(self, actor, critic, time_step, device, gpu_id):
    super().__init__()
    
    self.actor = actor
    self.actor_optimizer = MADGRAD(actor.parameters() ,lr = 1e-3)
    
    self.critic = critic
    self.critic_optimizer = MADGRAD(critic.parameters() ,lr = 1e-3)
    
    self.time_step = time_step
    self.device = device
    self.gpu_id = gpu_id
    
    
  def critic_forward(self, state, action, eval = False):
    
    # train or eval model
    if eval == True:
      self.critic.eval()
    else:
      self.critic.train()
    
    # initialize hx,cx
    hx = torch.zeros((state.size()[0], self.actor.hidden_size), device = self.device)
    cx = torch.zeros((state.size()[0], self.actor.hidden_size), device = self.device)
    
    # get actions shape(batch_size,time_step,action_size)
    values = torch.FloatTensor()
    
    # 對時間點遍歷
    for t in range(self.time_step):
      # 根據當前t時刻狀態,動作,hx,cx當輸入得到,v(產出),更新hx,cx
      v, hx, cx = self.critic(torch.cat((state[:,t,:], action[:,t,:]), dim = 1), hx, cx)
      values = torch.cat((values, v.cpu()), dim = 1)
      
    # 一系列產出(Batch,Time,Features)
    values = values.reshape(state.size()[0], self.time_step, -1)
    
    # 只取最後一個時間點
    return values[:,-1,:] # return last time_step
  
  
  def actor_forward(self, state, request, eval = False):
    '''
    # 給定狀態(不可控)和需求 輸出一系列動作
    '''
    # train or eval model
    if eval == True:
      self.actor.eval()
    else:
      self.actor.train()
      
    
    # initialize hx,cx
    hx = torch.zeros((state.size()[0], self.actor.hidden_size), device = self.device)
    cx = torch.zeros((state.size()[0], self.actor.hidden_size), device = self.device)
    
    # get actions shape(batch_size,time_step,action_size)
    actions = torch.FloatTensor()
    
    # 對一定時間長度進行遍歷
    for t in range(self.time_step):
      # 在t時刻 根據t時刻的狀態,需求以及短期記憶hx,長期記憶cx當作輸入,得到輸出a(動作),更新後的短期記憶hx,更新後的長期記憶cx
      a, hx, cx = self.actor(torch.cat((state[:,t,:], request), dim = 1), hx, cx)
      actions = torch.cat((actions, a.cpu()), dim = 1)
    # 一系列動作 (Batch,Time,Features)
    actions = actions.reshape(state.size()[0], self.time_step, -1)
    
    return actions
  
  def critic_loss_fun(self, state, action, value):
    '''
    x: [state,action]
    y: [value]
    '監督式學習'
    '''
    # self.critic.train()
    # self.actor.eval()
    value_hat = self.critic_forward(state, action)
      
    # 預測值跟label的平方差愈小愈好
    loss = ((value_hat-value)**2).mean()
    loss.backward()
    self.critic_optimizer.step()
    self.critic_optimizer.zero_grad()
  
    return loss.item()
  
  
  def actor_loss_fun(self, state, request, action_size):
  
    actions = self.actor_forward(state, request)
    value_hat = self.critic_forward(state, actions.cuda(device = self.gpu_id))
    
    # 首先預測值跟需求的平方差愈小愈好
    loss1 = ((request.cpu() - value_hat)**2).mean()
  
    # 再來避免"變異數太大(時間維度上)" 因為盤控人員不可能突然調太多
    loss2 = actions.reshape(-1, self.time_step, action_size).std(axis = 1)
    loss2 = loss2.sum(axis = -1).mean(axis = 0)
  
    loss = loss1 + loss2
    
    loss.backward()
      
    self.actor_optimizer.step()
    self.actor.zero_grad()
    
    return loss.item()
    
