from yahoofinancials import YahooFinancials

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import pandas as pd
import os
import math
import datetime
import art

from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import gradcheck
import time


class Env(): #ici il faut interagir avec l'environnement
        def __init__(self, list_stock, date, max_length):
                super(Env, self).__init__()
                self.list_stock = list_stock
                self.date = date

                self.data = self.collect_data(self.list_stock, self.date)

                #preparing the date for env
                self.initial_state = self.process_data(self.data, max_length)
                #print(self.initial_state)

        def get_data(self):
               return self.initial_state


        
        def collect_data(self, list_stock, date): #date = ["2010-06-28", "2010-07-05"]
                                                #list_stock = ["^DJI", "TSLA", "AAPL", "MSFT", "GOOG", "AMZN"] c'est un exemple (j'ai vérifié que les données sont fiables et c'est resistant aux dates de week end)
                print("starting collecting data from Yahoo")
                yahoo_financials = YahooFinancials(list_stock)
                data=yahoo_financials.get_historical_price_data(start_date = date[0], end_date = date[1], time_interval = "daily")
                
                print("data downloaded, let's process that!")
                
                data_for_df = {"Date": []}
                all_dates = set()
                
                for stock in list_stock:
                    data_for_df[stock] = []
                    stock_data = data.get(stock, {}).get("prices", [])
                    stock_data = stock_data[::-1]
                    for record in stock_data:
                        date = record.get("formatted_date")
                        if date not in all_dates:
                            all_dates.add(date)
                            data_for_df["Date"].append(date)
                        avg_price = (record['low']+ record['high'])/2
                        data_for_df[stock].append(avg_price)


                max_length = max(len(data_for_df["Date"]), max(len(data_for_df[stock]) for stock in list_stock))
                for stock in list_stock:
                      while len(data_for_df[stock]) < max_length:
                            data_for_df[stock].append(np.nan)


                df = pd.DataFrame(data_for_df)
                df.set_index('Date', inplace=True)

                current_file_path = os.path.abspath(__file__)
                current_directory = os.path.dirname(current_file_path)
                chemin_fichier_csv = os.path.join(current_directory, 'nom_fichier.csv')

                df.to_csv(chemin_fichier_csv, index=True)

                numpy_array = df.to_numpy()
                numpy_array = numpy_array[::-1].copy()

                print("data collected")
                return numpy_array #ca sort un numpy array ou les colonnes représentes les actifs et les lignes les dates, ca commence de la date la plus récente chiffre eleveé  vers la plus ancienne chiffres bas
        
        

        def process_data(self, data, max_length): #make data start with t = max_length to make model be able to have env data
                
                
                processed_data = []
                if max_length + 1 > len(data):
                       raise ValueError("la longueur de la file ne peut pas être plus grande que la longueur des données +1")
                
                data_tensor = torch.tensor(data, dtype=torch.float32)

                for i in range(max_length, len(data)):
                        state = data_tensor[i - max_length:i]
                        next_state = data_tensor[i]

                        processed_data.append((state, next_state))

                        if i + max_length + 1 >= len(data):
                               break
                
                
                
                return processed_data


        def reward(y_return, weights):
                weights = torch.unsqueeze(weights, 1)
                meanReturn = torch.unsqueeze(torch.mean(y_return, axis=1), 2)
                covmat = torch.Tensor([np.cov(batch.cpu().T, ddof=0) for batch in y_return]).to("cpu")
                portReturn = torch.matmul(weights, meanReturn)
                portVol = torch.matmul(
                        weights, torch.matmul(covmat, torch.transpose(weights, 2, 1))
                )
                objective = (portReturn * 12 - 0.02) / (torch.sqrt(portVol * 12))
                return -objective.mean()
        
        def volatility_portfolio(self, prices, weight):
                # Convertir les prix en rendements
                returns = (prices[1:] - prices[:-1]) / prices[:-1]
                
                portfolio_returns = torch.matmul(returns, weight)
                portfolio_volatility = torch.std(portfolio_returns) * torch.sqrt(torch.tensor(252.0))
                print(portfolio_volatility)
                
                # Remplacement des NaN par la moyenne de chaque colonne
                nan_mask = torch.isnan(returns)
                col_mean = torch.nanmean(returns, dim=0, keepdim=True)
                returns = torch.where(nan_mask, col_mean, returns)
                
                mean_centered_data = returns - torch.mean(returns, dim=0, keepdim=True)
                
                cov_matrix = torch.mm(mean_centered_data.t(), mean_centered_data) / (returns.size(0) - 1)
                
                #test
                weights_col = weight.unsqueeze(1)
    
                # Produit matriciel (q^T * Σ * q)
                ann_portfolio_variance = torch.matmul(torch.matmul(weight, cov_matrix), weights_col)


                return torch.sqrt(ann_portfolio_variance * 252)



    
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length):
        super(PositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Create the positional encodings once in log space
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe = torch.zeros(max_length, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class transformer_model(nn.Module):
        def __init__(self, embedding_dim, max_length, nhead,  num_layers, dropout, num_stocks):
                super(transformer_model, self).__init__()

                self.data_embedding = nn.Linear(num_stocks, embedding_dim)
                # Positional encoding
                
                self.positional_encoding = PositionalEncoding(embedding_dim, max_length)

                

                #transformer
                self.transformer_encoder = nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(
                                d_model=embedding_dim,
                                nhead=nhead,
                                dropout=dropout,
                                batch_first=True
                                ),
                num_layers=num_layers
        )

                # Output layer
                self.output_layer = nn.Linear(embedding_dim * max_length, num_stocks)
                self.softmax = nn.Softmax(dim=-1)


        def forward(self, x):
                #reshape input to add the embedding dimension
                # Handle NaNs
                #x = torch.nan_to_num(x, nan=0.0)

                x = self.data_embedding(x)

                x_embedded = self.positional_encoding(x)

                #transformer forward
                transformer_out = self.transformer_encoder(x_embedded) #using for encoder and decoder input

                # Flatten transformer output to (batch_size, max_length * embedding_dim)
                flattened = transformer_out.view(transformer_out.size(0), -1)

                # Project to q_dim
                output = self.output_layer(flattened)

                
                output = self.softmax(output)

                return output

class transformer_model3(nn.Module):
    def __init__(self, embedding_dim, max_length, nhead,  num_layers, dropout, num_stocks):
        super(transformer_model3, self).__init__()
        self.data_embedding = nn.Linear(num_stocks, embedding_dim)
        self.linear = nn.Linear(embedding_dim * max_length, num_stocks)
        self.linear2 = nn.Linear(num_stocks, num_stocks)
        
        # Softmax layer
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.data_embedding(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.softmax(x)
        return x


class transformer_model2(nn.Module):
    def __init__(self, embedding_dim, max_length, nhead,  num_layers, dropout, num_stocks):
        super(transformer_model2, self).__init__()
        self.data_embedding = nn.Linear(num_stocks, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, max_length)
        # Une couche linéaire simple pour remplacer le transformer


        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=4,
                nhead=2,
                dropout=0.1,
                batch_first=True
                                ),
            num_layers=2
        )
        
        self.linear = nn.Linear(embedding_dim * max_length, num_stocks)
        
        # Softmax layer
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Pas de gestion de NaN ici pour simplifier, mais vous pouvez l'ajouter si nécessaire
        x = self.data_embedding(x)

        x1 = self.positional_encoding(x)
        transformer_out = self.transformer_encoder(x1) #using for encoder and decoder input
        print(x1.shape)


        x_reshaped = x.view(x.size(0), -1)
        print(transformer_out.shape)
        x = self.linear(x_reshaped)
        x = self.softmax(x)
        return transformer_out



class Transformer(nn.Module):
       def __init__(self, num_epochs, lr, batch_size, embedding_dim, dropout, max_length, nhead, num_layers, num_stocks):
                super(Transformer, self).__init__()
                self.num_epochs = num_epochs
                self.lr = lr
                self.batch_size = batch_size
                self.dropout = dropout
                self.embedding_dim = embedding_dim
                self.max_length = max_length
                self.nhead = nhead
                self.num_layers = num_layers

                self.model = transformer_model(self.embedding_dim, max_length, nhead, num_layers, dropout, num_stocks)
                self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
                
                #créer un nom de run unique
                run_name = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                log_dir = os.path.join('runs', run_name)

                self.writer = SummaryWriter(log_dir)
       
       def train(self):
                # Calculate number of total parameters in the model
                total_params = sum(p.numel() for p in self.model.parameters())
                print(f"Total number of parameters: {total_params}")

                # Data for the env
                date = ["2010-06-28", "2019-08-05"]
                list_stock = ["^DJI", "TSLA", "AAPL", "MSFT", "GOOG", "AMZN"]

                # Creating the env for training
                env = Env(list_stock, date, self.max_length)
                data = env.get_data()
                
                #preparation of training data and eval data
                ratio_eval = 0.2
                point_decoupage = int(len(data) * (1-ratio_eval))

                training = data[:point_decoupage]
                
                eval = data[point_decoupage:]
                #eval reprsente 20% des données (dans l'ordre temporel)
                self.gradient_accumulator = 0

                for epoch in tqdm(range(num_epochs), desc="Epoch", unit="epoch", leave=True):
                        epoch_loss = 0
                        for step in range(len(training)):
                                state = training[step][0].detach()
                                next_state = training[step][1].detach()

                                def contains_nan(tensor):
                                        """Vérifie si le tenseur contient une ou plusieurs valeurs NaN."""
                                        return torch.isnan(tensor).any() or (tensor == 0).any()
                                
                                if contains_nan(state) or contains_nan(next_state):
                                        # Passer à l'itération suivante si un NaN est détecté
                                        print("pass")
                                        continue
                                
                                state = state.unsqueeze(0)
                                #state = torch.nan_to_num(state, nan=1)
                                #next_state = torch.nan_to_num(next_state, nan=1)

                                q = self.model(state)

                                
                                
                                reward = env.reward(state, q)
                                print(reward)

                                loss = - torch.mean(torch.log(q) * reward.detach())
                                print(loss)
                                #time.sleep(1)
                                print(state)
                                print(next_state)
                                #print(action_value)

                                self.gradient_accumulator += 1
                                #loss = torch.clamp(loss, min=-100, max=100)
                                
                                loss.backward()  # Accumuler les gradients dans .grad

                                if self.gradient_accumulator % 30 == 0:
                                # Tous les 10 steps, effectuer la mise à jour
                                        self.optimizer.step()
                                        self.optimizer.zero_grad()  # Réinitialiser les gradients après la mise à jour
                                        self.gradient_accumulator = 0  # Réinitialiser le compteur de gradients

                                

                                epoch_loss += loss.item()

                        train_loss = epoch_loss / len(training)
                        self.writer.add_scalar('Loss/train', train_loss, epoch)

                        #print(f"Epoch {epoch}/{self.num_epochs}, Loss: {loss:.4f}")

                        
                        if epoch % 5 == 0:
                                #test every 50 epochs
                                self.model.eval()
                                with torch.no_grad():
                                       eval_loss = 0
                                       eval_sharpe = 0
                                       for eval_step in range(len(eval)):
                                              state = eval[eval_step][0]
                                              next_state = eval[eval_step][1]
                                              state = state.unsqueeze(0)
                                              action_value = self.model(state)
                                              reward = env.reward(action_value, state, next_state)
                                              
                                              loss = -reward
                                              loss = torch.clamp(loss, min = -3, max = 3)
                                              eval_loss += loss.item()

                                              eval_sharpe += reward.item()
                                self.model.train()
                                
                                eval_loss = eval_loss / len(eval)
                                eval_sharpe = - eval_loss

                                self.writer.add_scalar('Loss/eval', eval_loss, epoch)
                                self.writer.add_scalar('Sharpe/eval', eval_sharpe, epoch)

                                #save every 50 epochs

                                script_dir = os.path.dirname(os.path.abspath(__file__))
                                saves_dir = os.path.join(script_dir, 'saves')
                                if not os.path.exists(saves_dir):
                                       os.makedirs(saves_dir)

                                checkpoint_path = os.path.join(saves_dir, f'checkpoint_epoch_{epoch}.pth')


                                checkpoint = {
                                        'epoch': epoch,
                                        'model_state_dict': self.model.state_dict(),
                                        'optimizer_state_dict': self.optimizer.state_dict(),
                                }
                                torch.save(checkpoint, checkpoint_path)
                                #print(f"Checkpoint saved at {checkpoint_path}")

                self.writer.close()
                                





num_epochs = 2000
lr = 0.001
gamma = 0.99
batch_size = 16
embedding_dim = 20
dropout = 0.1
max_length = 14
nhead = 2
num_layers = 2
num_stocks = 6

print("\033[H\033[J", end="")

art.tprint("RMBP", "rnd-xlarge") 
art.tprint(".inc")

transformer = Transformer(num_epochs=num_epochs, lr=lr, batch_size=batch_size, embedding_dim=embedding_dim, dropout=dropout, max_length=max_length, nhead=nhead, num_layers=num_layers, num_stocks=num_stocks)

transformer.train()



#writer = SummaryWriter('runs/experiment_1')
