import pandas as pd
import pandas_datareader.data as web
import datetime
import numpy as np
import matplotlib.pyplot as plt
import warnings
import yfinance as yf
import statsmodels.api as sm
import math
import seaborn as sns
import time 
import random
from scipy.interpolate import LSQUnivariateSpline as Spline
import pandas as pd
import yfinance as yf
import requests
import csv


def get_std_ticker(symbol, s_date, e_date):
    warnings.filterwarnings("ignore")
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey=GBF9XORBS6GPC2IH'
    r = requests.get(url)
    data = r.json()
    dfstd = data["Time Series (Daily)"]
    dfstd = pd.DataFrame.from_dict(dfstd, orient="index")
    dfstd.columns= ["Open", 'High', 'Low', 'Close', 'Volume']
    dfstd = dfstd.astype({"Open": float, "High": float, "Low": float, "Close": float, "Volume": float})
    dfstd.reset_index(inplace = True)
    dfstd = dfstd.rename(columns={"index": "Date"})
    dfstd = dfstd.sort_index(ascending=False).reset_index(drop=True)
    
    return dfstd

def append_stock(dataset, symbol, s_date, e_date, std_tick_dict):
    warnings.filterwarnings("ignore")
    # Fetch historical data
    df = std_tick_dict[f'df{symbol}'] #####need to have run get_std_ticker (NEXT TIME- IMPLEMENT IF STATEMENT TO CHECK IF THERE)

    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df["Date"] >= s_date]
    df = df[df["Date"] <= e_date]
    df['daten'] = df['Date'].apply(lambda x: int(x.timestamp()))

    df['daten'] = df['daten']/86400
    df = df.drop(index=df.index[:4])
    df['timedif'] = df['daten'].apply(lambda x: x-df['daten'].iloc[0])
    df['timedif'] = df['timedif'].astype('int')
    empty = pd.DataFrame({'timedif': range(0,max(df['timedif'].astype('int')))})
    empty['dups'] = empty['timedif']
    jdf = pd.merge(df, empty, on='timedif', how='outer')
    #print(jdf.head(15))
    fdf = jdf.sort_values(by = 'dups', ascending = True)
    df = fdf.fillna(method='ffill')
    df['drop'] = 'drop'
    #print(df.tail())
    i = 0
    for x in range(len(df)):
        i+=1
        if i == 1:
            df['drop'].iloc[x] = 'keep'
        elif i == 12:
            df['drop'].iloc[x] = 'keep'
        elif i == 14:
            i = 0

    df = df[df['drop'] == 'keep']
    df['OpenD'] = np.nan
    df['OpenDn'] = np.nan
    df['CloseD'] = np.nan
    df['CloseDn'] = np.nan
    df['td1'] = np.nan
    df['td2'] = np.nan
    df['OPrice'] = np.nan
    df['CPrice'] = np.nan

    for x in range(len(df)-1):
        if (df['timedif'].iloc[x] + 11) == (df['timedif'].iloc[x+1]):
            df['OpenD'].iloc[x] = df['Date'].iloc[x]
            df['OpenDn'].iloc[x] = df['daten'].iloc[x]
            df['CloseD'].iloc[x] = df['Date'].iloc[x+1]
            df['CloseDn'].iloc[x] = df['daten'].iloc[x+1]
            df['td1'].iloc[x] = df['timedif'].iloc[x]
            df['td2'].iloc[x] = df['timedif'].iloc[x+1]
            df['OPrice'].iloc[x] = df['Open'].iloc[x]
            df['CPrice'].iloc[x] = df['Close'].iloc[x+1]
    df.loc[df.index[-1], ['OPrice', 'CPrice']] = 999
    df.loc[df.index[-1], ['OpenD', 'OpenDn', 'CloseD', 'CloseDn', 'td1', 'td2']] = pd.to_datetime('2099-01-01')
    #print(df.tail())
    df = df.dropna()
    df['chg'] = (df['CPrice']-df['OPrice'])/df['OPrice']
    df['pchg'] = df['chg'].shift(1)

    #claim inds and sev - 10 percent
    df['gt10p'] = 0
    df['gt10n'] = 0
    df['sev10p'] = 0
    df['sev10n'] = 0
    for x in range(len(df)):
        if df['chg'].iloc[x] > 0.1:
            df['gt10p'].iloc[x] = 1
            df['sev10p'].iloc[x] = df['chg'].iloc[x] - 0.1
        if df['chg'].iloc[x] < -0.1:
            df['gt10n'].iloc[x] = 1
            df['sev10n'].iloc[x] = -(df['chg'].iloc[x] + 0.1) 

    #claim inds and sev - 5 percent
    df['gt5p'] = 0
    df['gt5n'] = 0
    df['sev5p'] = 0
    df['sev5n'] = 0
    for x in range(len(df)):
        if df['chg'].iloc[x] > 0.05:
            df['gt5p'].iloc[x] = 1
            df['sev5p'].iloc[x] = df['chg'].iloc[x] - 0.05
        if df['chg'].iloc[x] < -0.05:
            df['gt5n'].iloc[x] = 1
            df['sev5n'].iloc[x] = -(df['chg'].iloc[x] + 0.05) 

    #prev calculations
    df['p1sev5p'] = df['sev5p'].shift(1)
    df['p2sev5p'] = df['sev5p'].shift(2)
    df['p3sev5p'] = df['sev5p'].shift(3)

    df['p1sev5n'] = df['sev5n'].shift(1)
    df['p2sev5n'] = df['sev5n'].shift(2)
    df['p3sev5n'] = df['sev5n'].shift(3)

    df['p1gt5p'] = df['gt5p'].shift(1)
    df['p2gt5p'] = df['gt5p'].shift(2)
    df['p3gt5p'] = df['gt5p'].shift(3)

    df['p1gt5n'] = df['gt5n'].shift(1)
    df['p2gt5n'] = df['gt5n'].shift(2)
    df['p3gt5n'] = df['gt5n'].shift(3)

    #previous claim counts total over last 3 periods 
    df['p6wkclaimsp'] = df['p1gt5p'] + df['p2gt5p'] + df['p3gt5p']
    df['p6wkclaimsn'] = df['p1gt5n'] + df['p2gt5n'] + df['p3gt5n']

    #Bin target
    df['bsev5p'] = 0
    df['bsev5n'] = 0 

    def binned(sev):
        if sev == 0:
            return 0
        elif sev <= 0.03:
            return 1
        elif sev <= 0.07:
            return 2
        else:
            return 3

    # Create a new categorical column for the groups
    df['bsev5p'] = df['sev5p'].apply(binned)
    df['bsev5n'] = df['sev5n'].apply(binned)
        
        
    

    ##############                           #########################################################################
    ############## Price and Volume Variance #########################################################################
    ##############                           #########################################################################
    
    #### joining back to earlier df to make the calculations ####

    #first iterate and make new columns on the intermediate dataframe
    jdf['Volchg'] = jdf['Volume'].pct_change()
    jdf['Closechg'] = jdf['Close'].pct_change()

    #print("now jdf:")
    #print(jdf.head(20))

    df['closevar'] = 0
    df['volvar'] = 0

    for row in range(len(df)-1):
        tnum = df['timedif'].iloc[row+1]
        #print("tnum is:", tnum)
        tndf = jdf[jdf['timedif'] < tnum]
        tndf = tndf.dropna(subset = ['Volume'])
        #print(tndf.head(20))
        cvar = tndf['Closechg'].var()
        vvar = tndf['Volchg'].var()
        #print("cvar is:", cvar)
        df['closevar'].iloc[row] = cvar
        df['volvar'].iloc[row] = vvar

        #drop those rows
        jdf = jdf[jdf['timedif'] >= tnum]

    #t-1
    df['pclosevar'] = df['closevar'].shift(1)
    df['pvolvar'] = df['volvar'].shift(1)
    df['p2volvar'] = df['volvar'].shift(2)
    df['p2closevar'] = df['closevar'].shift(2)
    df['pvolvarchg'] = df['pvolvar'].pct_change()
    df['pclosevarchg'] = df['pclosevar'].pct_change()



    ##join data
    df.columns = symbol + '_' + df.columns

    data = dataset.merge(df, left_on= 'timedif', right_on = symbol + '_' + 'timedif', how = 'left')

    return data


def getData(symbol, s_date, e_date, std_tick_dict):

    warnings.filterwarnings("ignore")
    # Fetch historical data
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey=GBF9XORBS6GPC2IH'
    r = requests.get(url)
    data = r.json()
    df = data["Time Series (Daily)"]
    df = pd.DataFrame.from_dict(df, orient="index")
    df.columns= ["Open", 'High', 'Low', 'Close', 'Volume']
    df = df.astype({"Open": float, "High": float, "Low": float, "Close": float, "Volume": float})
    df.reset_index(inplace = True)
    df = df.rename(columns={"index": "Date"})
    df = df.sort_index(ascending=False).reset_index(drop=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df["Date"] >= s_date]
    df = df[df["Date"] <= e_date]
    df['daten'] = df['Date'].apply(lambda x: int(x.timestamp()))

    df['daten'] = df['daten']/86400
    df = df.drop(index=df.index[:4])
    df['timedif'] = df['daten'].apply(lambda x: x-df['daten'].iloc[0])
    df['timedif'] = df['timedif'].astype('int')
    empty = pd.DataFrame({'timedif': range(0,max(df['timedif'].astype('int')))})
    empty['dups'] = empty['timedif']
    jdf = pd.merge(df, empty, on='timedif', how='outer')
    #print(jdf.head(15))
    fdf = jdf.sort_values(by = 'dups', ascending = True)
    df = fdf.fillna(method='ffill')
    df['drop'] = 'drop'
    #print(df.tail())
    i = 0
    for x in range(len(df)):
        i+=1
        if i == 1:
            df['drop'].iloc[x] = 'keep'
        elif i == 12:
            df['drop'].iloc[x] = 'keep'
        elif i == 14:
            i = 0

    df = df[df['drop'] == 'keep']
    df['OpenD'] = np.nan
    df['OpenDn'] = np.nan
    df['CloseD'] = np.nan
    df['CloseDn'] = np.nan
    df['td1'] = np.nan
    df['td2'] = np.nan
    df['OPrice'] = np.nan
    df['CPrice'] = np.nan

    for x in range(len(df)-1):
        if (df['timedif'].iloc[x] + 11) == (df['timedif'].iloc[x+1]):
            df['OpenD'].iloc[x] = df['Date'].iloc[x]
            df['OpenDn'].iloc[x] = df['daten'].iloc[x]
            df['CloseD'].iloc[x] = df['Date'].iloc[x+1]
            df['CloseDn'].iloc[x] = df['daten'].iloc[x+1]
            df['td1'].iloc[x] = df['timedif'].iloc[x]
            df['td2'].iloc[x] = df['timedif'].iloc[x+1]
            df['OPrice'].iloc[x] = df['Open'].iloc[x]
            df['CPrice'].iloc[x] = df['Close'].iloc[x+1]
    df.loc[df.index[-1], ['OPrice', 'CPrice']] = 999
    df.loc[df.index[-1], ['OpenD', 'OpenDn', 'CloseD', 'CloseDn', 'td1', 'td2']] = pd.to_datetime('2099-01-01')
    #print(df.tail())
    df = df.dropna()
    df['chg'] = (df['CPrice']-df['OPrice'])/df['OPrice']
    df['pchg'] = df['chg'].shift(1)

    #claim inds and sev - 10 percent
    df['gt10p'] = 0
    df['gt10n'] = 0
    df['sev10p'] = 0
    df['sev10n'] = 0
    for x in range(len(df)):
        if df['chg'].iloc[x] > 0.1:
            df['gt10p'].iloc[x] = 1
            df['sev10p'].iloc[x] = df['chg'].iloc[x] - 0.1
        if df['chg'].iloc[x] < -0.1:
            df['gt10n'].iloc[x] = 1
            df['sev10n'].iloc[x] = -(df['chg'].iloc[x] + 0.1) 

    #claim inds and sev - 5 percent
    df['gt5p'] = 0
    df['gt5n'] = 0
    df['sev5p'] = 0
    df['sev5n'] = 0
    for x in range(len(df)):
        if df['chg'].iloc[x] > 0.05:
            df['gt5p'].iloc[x] = 1
            df['sev5p'].iloc[x] = df['chg'].iloc[x] - 0.05
        if df['chg'].iloc[x] < -0.05:
            df['gt5n'].iloc[x] = 1
            df['sev5n'].iloc[x] = -(df['chg'].iloc[x] + 0.05) 

    #prev calculations
    df['p1sev5p'] = df['sev5p'].shift(1)
    df['p2sev5p'] = df['sev5p'].shift(2)
    df['p3sev5p'] = df['sev5p'].shift(3)

    df['p1sev5n'] = df['sev5n'].shift(1)
    df['p2sev5n'] = df['sev5n'].shift(2)
    df['p3sev5n'] = df['sev5n'].shift(3)

    df['p1gt5p'] = df['gt5p'].shift(1)
    df['p2gt5p'] = df['gt5p'].shift(2)
    df['p3gt5p'] = df['gt5p'].shift(3)

    df['p1gt5n'] = df['gt5n'].shift(1)
    df['p2gt5n'] = df['gt5n'].shift(2)
    df['p3gt5n'] = df['gt5n'].shift(3)

    #previous claim counts total over last 3 periods 
    df['p6wkclaimsp'] = df['p1gt5p'] + df['p2gt5p'] + df['p3gt5p']
    df['p6wkclaimsn'] = df['p1gt5n'] + df['p2gt5n'] + df['p3gt5n']

    #Bin target
    df['bsev5p'] = 0
    df['bsev5n'] = 0 

    def binned(sev):
        if sev == 0:
            return 0
        elif sev <= 0.03:
            return 1
        elif sev <= 0.07:
            return 2
        else:
            return 3

    # Create a new categorical column for the groups
    df['bsev5p'] = df['sev5p'].apply(binned)
    df['bsev5n'] = df['sev5n'].apply(binned)
        
        
    

    ##############                           #########################################################################
    ############## Price and Volume Variance #########################################################################
    ##############                           #########################################################################
    
    #### joining back to earlier df to make the calculations ####

    #first iterate and make new columns on the intermediate dataframe
    jdf['Volchg'] = jdf['Volume'].pct_change()
    jdf['Closechg'] = jdf['Close'].pct_change()

    #print("now jdf:")
    #print(jdf.head(20))

    df['closevar'] = 0
    df['volvar'] = 0

    for row in range(len(df)-1):
        tnum = df['timedif'].iloc[row+1]
        #print("tnum is:", tnum)
        tndf = jdf[jdf['timedif'] < tnum]
        tndf = tndf.dropna(subset = ['Volume'])
        #print(tndf.head(20))
        cvar = tndf['Closechg'].var()
        vvar = tndf['Volchg'].var()
        #print("cvar is:", cvar)
        df['closevar'].iloc[row] = cvar
        df['volvar'].iloc[row] = vvar

        #drop those rows
        jdf = jdf[jdf['timedif'] >= tnum]

    #t-1
    df['pclosevar'] = df['closevar'].shift(1)
    df['pvolvar'] = df['volvar'].shift(1)
    df['p2volvar'] = df['volvar'].shift(2)
    df['p2closevar'] = df['closevar'].shift(2)
    df['pvolvarchg'] = df['pvolvar'].pct_change()
    df['pclosevarchg'] = df['pclosevar'].pct_change()

    
    ##############               #########################################################################
    ############## Earnings Date #########################################################################
    ##############               #########################################################################

    ##Upcoming Earnings##    
    CSV_URL = f'https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&symbol={symbol}&horizon=6month&apikey=GBF9XORBS6GPC2IH'

    with requests.Session() as s:
        download = s.get(CSV_URL)
        decoded_content = download.content.decode('utf-8')
        cr = csv.reader(decoded_content.splitlines(), delimiter=',')
        my_list = list(cr)

    columns = my_list[0]
    earns = pd.DataFrame(my_list[1:], columns=columns)
    earns.replace('', pd.NA, inplace=True)
    earns = earns.rename(columns={"reportDate": "Earnings Date"})
    earns["Earnings Date"] = pd.to_datetime(earns["Earnings Date"])
    earns['daten'] = earns['Earnings Date'].apply(lambda x: int(x.timestamp()))
    earns['daten'] = np.floor(earns['daten']/86400)

    #Hardcode 2012-01-09 as 0
    zdate = pd.to_datetime('2012-01-09')
    zdate = int(zdate.timestamp())
    zdate = zdate/86400
        
    earns['timedif'] = earns['daten'].apply(lambda x: x-zdate)
    earns['timedif'] = earns['timedif'].astype('int')

    dfnew = earns[['Earnings Date', 'daten', 'timedif']].copy()

    ##Past Earnings##
    url = f'https://www.alphavantage.co/query?function=EARNINGS&symbol={symbol}&apikey=GBF9XORBS6GPC2IH'
    r = requests.get(url)
    data = r.json()
    earnold = pd.DataFrame(data['quarterlyEarnings'])

    earnold = earnold.rename(columns={"reportedDate": "Earnings Date"})
    earnold["Earnings Date"] = pd.to_datetime(earnold["Earnings Date"])
    earnold['daten'] = earnold['Earnings Date'].apply(lambda x: int(x.timestamp()))
    earnold['daten'] = np.floor(earnold['daten']/86400)
    zdate = pd.to_datetime('2012-01-09')
    zdate = int(zdate.timestamp())
    zdate = zdate/86400
        
    earnold['timedif'] = earnold['daten'].apply(lambda x: x-zdate)
    earnold['timedif'] = earnold['timedif'].astype('int')

    dfprev= earnold[['Earnings Date', 'daten', 'timedif']].copy()


    dfearns = pd.concat([dfprev,dfnew])
    dfearns = dfearns.sort_values(by = 'Earnings Date', ascending=False)

    
    def closest_greater(row):
        greater_values = dfearns[dfearns['timedif'] > row['timedif']]['timedif']
        if len(greater_values) > 0:
            closest_greater_value = greater_values.iloc[-1]
            return closest_greater_value
        else:
            return None

    df['earntimedif'] = df.apply(closest_greater, axis=1)
    df['DaysTilEarn'] = df['earntimedif'] - df['timedif']

    df['Stock'] = symbol
    
    if max(df['DaysTilEarn']) > 200:
        print('EARNINGS_ERROR')
    #df = df[df['DaysTilEarn'] <= 150]

    #capping
    df['sev5pcap'] = df['sev5p'].clip(upper=0.25)


    # Sector
    Tech = ['MSFT', 'META', 'AMZN', 'NFLX', 'NVDA', 'GOOGL', 'INTC', 'TSLA', 'ORCL', 'AMD', 'CSCO', 'AAPL', 'TXN', 'MU', 'QCOM', 'ADBE', 'CRM']
    Finance = ['GS', 'BAC', 'JPM', 'C', 'MS', 'BLK', 'V', 'MA', 'AXP', 'PGR', 'PYPL']
    Healthcare = ['LLY', 'JNJ', 'MRK', 'PFE', 'GILD', 'BMY', 'UNH', 'MDT']
    Energy = ['XOM', 'CVX', 'COP']
    Consumer = ['WMT', 'COST', 'HD', 'KO', 'MCD', 'NKE', 'SBUX', 'MDLZ', 'MO', 'CMG', 'CVS', 'TGT']
    Telecom = ['VZ', 'CMCSA', 'T']
    
    
    df['Sector'] = ''

    if symbol in Tech:
        df['Sector'] = 'Tech'
    elif symbol in Finance:
        df['Sector'] = 'Finance'
    elif symbol in Healthcare:
        df['Sector'] = 'Healthcare'
    else:
        df['Sector'] = 'Unknown'


    df = append_stock(df, 'SPY', s_date, e_date, std_tick_dict)
    df = append_stock(df, 'USO', s_date, e_date, std_tick_dict)
    df = append_stock(df, 'VIXY', s_date, e_date, std_tick_dict)
    df = append_stock(df, 'GLD', s_date, e_date, std_tick_dict)

    return df






def binner(var):
    quantiles = var.quantile([0.25,0.5,0.75])
    def assign(value):
        if value < quantiles[0.25]:
            return 0
        elif value < quantiles[0.5]:
            return 1
        elif value < quantiles[0.75]:
            return 2
        elif value > quantiles[0.75]:
            return 3
        else:
            return 0

    return var.apply(assign)





def woe_calc(column, target):
    # Combine the predictor column and target into a single DataFrame
    df = pd.DataFrame({'predictor': column, 'target': target})
    
    # Calculate the overall mean target value
    overall_mean = df['target'].mean()
    
    # Calculate the mean target value for each bin
    mean_target_per_bin = df.groupby('predictor')['target'].mean()
    
    # Calculate the WOE-like transformation (logarithm of the ratio of mean target to overall mean)
    woe_values = np.log(mean_target_per_bin / overall_mean)
    
    # Map the WOE values back to the original column
    transformed_column = column.map(woe_values)
    
    return transformed_column



def calculate_bin_accuracy(df, prediction_col, target_col):
    # Initialize an empty dictionary to store accuracy for each bin
    bin_accuracy = {}
    
    # Iterate over each bin value (0, 1, 2, 3)
    for bin_value in sorted(df[target_col].unique()):
        # Filter the DataFrame for the current bin
        bin_df = df[df[target_col] == bin_value]
        
        # Calculate accuracy for the current bin
        correct_predictions = (bin_df[prediction_col] == bin_value).sum()
        total_predictions = bin_df.shape[0]
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        # Store the accuracy in the dictionary
        bin_accuracy[bin_value] = accuracy
        
        # Print the accuracy for the current bin
        print(f"Accuracy for bin {bin_value}: {accuracy:.2f}")
    
    return bin_accuracy



def plot_relative_mean_target(predictor, target, df):
    # Calculate the mean target value for each bin
    mean_target = df.groupby(predictor)[target].mean().reset_index()
    
    # Calculate the overall mean target value
    overall_mean = df[target].mean()
    
    # Calculate the relative mean target by dividing by the overall mean
    mean_target['relative_mean_target'] = mean_target[target] / overall_mean
    
    # Calculate the percentage of observations for each bin
    pct_observations = df[predictor].value_counts(normalize=True).sort_index() * 100
    
    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Bar plot for the percentage of observations
    sns.barplot(x=pct_observations.index, y=pct_observations.values, color='lightblue', ax=ax1)
    ax1.set_ylabel('Percentage of Observations', color='blue')
    ax1.set_xlabel('Predictor Value')
    
    # Secondary y-axis for the relative mean target
    ax2 = ax1.twinx()  
    sns.lineplot(x=mean_target[predictor], y=mean_target['relative_mean_target'], marker='o', color='red', ax=ax2)
    ax2.set_ylabel('Relative Mean Target Value', color='red')
    ax2.axhline(1, color='black', linestyle='--')  # Add a line at y=1 (the overall mean)
    
    plt.title(predictor)
    plt.show()


def sevgroups2(sev):
    if sev == 0:
        return 0
    elif sev <= 0.04:
        return 1
    elif sev <= 0.08:
        return 2
    else:
        return 3