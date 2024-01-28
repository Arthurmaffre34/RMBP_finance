from yahoofinancials import YahooFinancials

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym

from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter



class Env(gym.Env): #ici il faut interagir avec l'environnement
        def __init__(self):
                super(Env, self).__init__()
                self.counter = 0

                self.data = self.collect_data()
                self.max_length = 20
                self.state = []
                self.w = 100000 #set amound of money agent have
                self.history = []
                self.rendements = []


        
        def collect_data():
                yahoo_financials = YahooFinancials('^DJI')
                data=yahoo_financials.get_historical_price_data("1992-01-01", "2023-09-18", "daily")
                prices = data['^DJI']['prices']

                data = []
                for price_data in prices:
                        high_value = price_data['high']
                        low_value = price_data['low']
                        average = (high_value + low_value) / 2
                        data.append(average)

                return data
        
        def process_data(self, counter, data, max_length, history): #make data start with t = max_length to make model be able to have env data
                states = []
                for i in range(max_length):
                        state = torch.tensor(data[i+counter:i+max_length+counter], dtype=torch.float32)
                        historys = torch.tensor(history[i+counter-max_length:+i+counter], dtype=torch.float32)
                        states.append(state, historys)

                next_state = torch.tensor(data[i+max_length+1+counter], dtype=torch.float32)
                processed_data = ((states, next_state))
                return processed_data
        
        def reward_function(self, q, action, state):
                next_p = state[1]
                prev_p = state[0][0]
                old_q = action

                q_difference = abs(q-old_q)

                tolerance = 0.01
                if q_difference > tolerance:
                      transaction_penalty = self.transaction_cost
                
                
                rendement = (((next_p * q) - (prev_p * q))/((prev_p * q)+self.w))*100

                self.rendements.append(rendement)
                self.rendements[-10:]

                variance = np.var(self.rendements)

                variance_penalty = self.variance_penalty_factor * variance

                reward = rendement - transaction_penalty - variance_penalty
                return reward
        
        def volatility_portfolio(data, weight): #ici on calcule la volatilité du portefeuille de manière parallélisée
                mean_centered_data = data - torch.mean(data, dim=1, keepdim=True)

                cov_matrix = torch.mm(mean_centered_data, mean_centered_data.t()) / (data.size(1) - 1)
                portfolio_variance = torch.dot(weight, torch.mv(cov_matrix, weight))

                return torch.sqrt(portfolio_variance)

        def reset(self):
              self.counter = 0
              initial_state = self.process_data(self.counter, self.data, self.max_length)
              self.counter += 1
              self.w = 100000
              self.history = []
              
              self.state = initial_state
              return initial_state
        
        def step(self, q, action):
                
                rendement = self.reward_function(q, action, state, self.w)
                
                self.w = (1+rendement) * self.w * q
                
                self.history.append((q, self.w))

                self.state = self.process_data(self.counter, self.data, self.max_length, self.history)


                if self.counter >= len(data)-1:
                        self.done = True

                reward = torch.tensor(rendement)
                state = torch.tensor(state[0])
                next_state = torch.tensor(state[1])
                done = torch.tensor(done)

                return state, next_state, reward, done




class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        #positional embedding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0,1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    


class Transformer(nn.Module):
        def __init__(self, embedding_dim, max_length, nhead,  num_layers, dropout):
                super(Transformer, self).__init__()

                #positional embedding
                self.positional_embedding = nn.Embedding(max_length, embedding_dim)

                self.data_embedding = nn.Linear(1, embedding_dim)

                #transformer
                self.transformer = nn.Transformer(
                        d_model=embedding_dim,
                        nhead=nhead,
                        num_encoder_layers=num_layers,
                        num_decoder_layers=num_layers,
                        dropout=dropout
                )

        def forward(self, x):
                #reshape input to add the embedding dimension
                x_embedded = self.data_embedding(x.unsqueeze(-1))

                #add positional embedding
                positions = torch.arange(len(x), device=x.device).unsqueeze(-1)

                x_embedded += self.positional_embedding(positions)

                #transformer forward
                transformer_out = self.transformer(x_embedded,x_embedded) #using for encoder and decoder input
                return transformer_out
        


def train(self):
            #calculate number of total parameters in the model
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"total number of parameters: {total_params}")

            #calculate the total number of batches
            total_batches = len(self.samples) // self.batch_size
            print(total_batches)
            if len(self.samples) % self.batch_size != 0:
                  total_batches += 1 #accounting for the last batch if it's not batch size

            if self.epoch_r != 0: #reprise en cas de load de modèle
                  print(f"reprise de l'entrainement à partir de {self.epoch_r}/{self.num_epochs}")

            
            for epochs in range(self.epoch_r, self.num_epochs):
                epoch_loss = 0.0
                for state, next_state, _ in self.batch_iter(samples=self.samples):
                        state = state.to(self.device)
                        next_state = next_state.to(self.device)
                        

                        state_value = self.model(state)

                        
                        next_state_value = self.model(next_state)

                        reward = self.rewardf(state_value, next_state)
                        reward = reward.to(self.device)
                        target_value = reward + self.gamma * next_state_value

                        model_loss = torch.mean((state_value - target_value) ** 2)
                        
                        self.model_optimizer.zero_grad()
                        model_loss.backward()
                        self.model_optimizer.step()
                        epoch_loss += model_loss.item()
                
                print(total_batches)
                avg_epoch_loss = epoch_loss / (total_batches * self.batch_size)

                print(f"Epoch {epochs}/{self.num_epochs}, Loss: {avg_epoch_loss:.4f}")

                self.writer.add_scalar("Epoch Loss", avg_epoch_loss, epochs)

                if epochs % 50 == 0:
                    checkpoint = {
                            'epoch': epochs,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.model_optimizer.state_dict(),
                            'loss': model_loss,
                      }
                    torch.save(checkpoint, f'checkpoint_epoch_{epochs}.pth')
            
            self.writer.close()




num_epochs = 5000
lr = 0.0001
gamma = 0.99
batch_size = 16
embedding_dim = 8
dropout = 0.1

transformer = Transformer(averages, num_epochs=num_epochs, lr=lr, gamma=gamma, batch_size=batch_size, embedding_dim=embedding_dim, dropout=dropout)

transformer.train()

#writer = SummaryWriter('runs/experiment_1')

def optimize_trading_dynamic(prices, forecasts, T, initial_capital):
      #define action
      BUY, SELL, HOLD = 0, 1, 2
      actions = [BUY, SELL, HOLD]

