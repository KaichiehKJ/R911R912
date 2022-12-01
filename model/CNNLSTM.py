
import torch
import torch.nn as nn


class cnn_lstm(nn.Module):
  def __init__(self, device, hidden_size):
    super().__init__()
    
    self.device = device
    self.hidden_size = hidden_size
    self.c1 = nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 1, stride = 1)
    self.lstm = nn.LSTM(
            input_size=19,
            hidden_size=self.hidden_size,
            num_layers=2
        )
    self.linear = nn.Linear(in_features=4*self.hidden_size, out_features=1)


  def forward(self, state, action):
    
    x = torch.cat((state, action), 2)
    x = x.reshape(len(x), 4, 1, 19)
    
    lstm_output = torch.zeros((32, 4, 1, 1*self.hidden_size), device = self.device)
    
    for i in range(len(x)):
      cnn_output = torch.tanh(self.c1(x[i,:,:,:]))
      lstm_output[i,:,:,:], _ = self.lstm(cnn_output)
    
    lstm_output = lstm_output.reshape(32, 4*self.hidden_size)
    y_pred = torch.sigmoid(self.linear(lstm_output))
    
    return y_pred 



if __name__=="__main__":
  
  model = cnn_lstm()


