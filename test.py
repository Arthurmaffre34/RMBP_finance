from yahoofinancials import YahooFinancials
import pandas as pd
import numpy as np
import os

date = ["2010-06-28", "2010-07-05"]
#list_stock = ["^DJI", "TSLA", "AAPL", "MSFT", "GOOG", "AMZN"]
list_stock = ["TSLA", "MSFT"]

def collect_data(list_stock, date):
                yahoo_financials = YahooFinancials(list_stock)
                data=yahoo_financials.get_historical_price_data(start_date = date[0], end_date = date[1], time_interval = "daily")
                
                
                
                data_for_df = {"Date": []}
                all_dates = set()
                
                for stock in list_stock:
                    data_for_df[stock] = []
                    stock_data = data.get(stock, {}).get("prices", [])
                    stock_data = stock_data[::-1]
                    print(data_for_df)
                    for record in stock_data:
                        date = record.get("formatted_date")
                        if date not in all_dates:
                            all_dates.add(date)
                            data_for_df["Date"].append(date)
                        avg_price = (record['low']+ record['high'])/2
                        print(record['low'])
                        print(record['high'])
                        data_for_df[stock].append(avg_price)


                max_length = max(len(data_for_df["Date"]), max(len(data_for_df[stock]) for stock in list_stock))
                for stock in list_stock:
                      while len(data_for_df[stock]) < max_length:
                            data_for_df[stock].append(np.nan)


                df = pd.DataFrame(data_for_df)
                df.set_index('Date', inplace=True)

                racine_projet = os.getcwd() 
                chemin_fichier_csv = os.path.join(racine_projet, 'nom_fichier.csv')

                df.to_csv(chemin_fichier_csv, index=True)

                print(df)
                numpy_array = df.to_numpy()

                print(numpy_array[0])

collect_data(list_stock, date)