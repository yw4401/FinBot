#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install yfinance


# In[ ]:





# In[ ]:





# In[2]:


import yfinance as yf

msft = yf.Ticker("MSFT")

# get all stock info
msft.info

# get historical market data
hist = msft.history(period="1mo")

# show meta information about the history (requires history() to be called first)
msft.history_metadata

# show actions (dividends, splits, capital gains)
msft.actions
msft.dividends
msft.splits
msft.capital_gains  # only for mutual funds & etfs

# show share count
msft.get_shares_full(start="2022-01-01", end=None)

# show financials:
# - income statement
msft.income_stmt
msft.quarterly_income_stmt
# - balance sheet
msft.balance_sheet
msft.quarterly_balance_sheet
# - cash flow statement
msft.cashflow
msft.quarterly_cashflow
# see `Ticker.get_income_stmt()` for more options

# show holders
msft.major_holders
msft.institutional_holders
msft.mutualfund_holders

# Show future and historic earnings dates, returns at most next 4 quarters and last 8 quarters by default. 
# Note: If more are needed use msft.get_earnings_dates(limit=XX) with increased limit argument.
msft.earnings_dates

# show ISIN code - *experimental*
# ISIN = International Securities Identification Number
msft.isin

# show options expirations
msft.options

# show news
msft.news

# get option chain for specific expiration
opt = msft.option_chain('2023-10-27')
# data available via: opt.calls, opt.puts


# In[3]:


msft.income_stmt.T


# In[14]:


msft.info


# In[2]:


import yfinance as yf

tickers = yf.Tickers('msft aapl goog')

# access each ticker using (example)
tickers.tickers['MSFT'].info
tickers.tickers['AAPL'].history(period="1mo")
tickers.tickers['GOOG'].actions


# In[15]:


msft.news


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


def fetch_data(ticker_symbol, period="1y"):
    """
    Fetch historical data for the given ticker symbol and period.
    """
    ticker = yf.Ticker(ticker_symbol)
    data = ticker.history(period=period)
    summary = ticker.info.get('longBusinessSummary', 'No summary available.')
    return data, summary

def plot_data(data, ticker_symbol, summary):
    """
    Plot the historical data and display the company's summary.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label='Close Price', color='blue')
    plt.title(f'{ticker_symbol} Stock Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Close Price (in currency)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print("\nCompany Summary:")
    print(summary)

def main():
    """
    Main function to interact with the user.
    """
    ticker_symbol = input("Enter the ticker symbol (e.g., MSFT): ").upper()
    period = input("Enter the period (e.g., 1d, 5d, 1mo, 1y, etc.): ")
    
    data, summary = fetch_data(ticker_symbol, period)
    plot_data(data, ticker_symbol, summary)
    
    # Provide a link to the Yahoo Finance page for the given ticker
    print("\nSource:")
    print(f"https://finance.yahoo.com/quote/{ticker_symbol}")

if __name__ == "__main__":
    main()


# In[5]:


def fetch_data(ticker_symbol, period="1y"):
    """
    Fetch historical data and KPIs for the given ticker symbol and period.
    """
    ticker = yf.Ticker(ticker_symbol)
    data = ticker.history(period=period)
    summary = ticker.info.get('longBusinessSummary', 'No summary available.')
    
    kpis = {
        "Market Cap": ticker.info.get("marketCap"),
        "P/E Ratio": ticker.info.get("trailingPE"),
        "Dividend Yield": ticker.info.get("dividendYield"),
        "52 Week High": ticker.info.get("fiftyTwoWeekHigh"),
        "52 Week Low": ticker.info.get("fiftyTwoWeekLow"),
        "Beta": ticker.info.get("beta"),
        "Volume": ticker.info.get("volume"),
        "Earnings Per Share (EPS)": ticker.info.get("trailingEps"),
        "Price-to-Book Ratio (P/B)": ticker.info.get("priceToBook"),
        "Return on Equity (ROE)": ticker.info.get("returnOnEquity"),
        "Debt-to-Equity Ratio": ticker.info.get("debtToEquity"),
        "Current Ratio": ticker.info.get("currentRatio"),
        "Operating Margin": ticker.info.get("operatingMargin")
    }
    
    return data, summary, kpis

def plot_data(data, ticker_symbol, summary, kpis):
    """
    Plot the historical data, display KPIs, and the company's summary.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label='Close Price', color='blue')
    plt.title(f'{ticker_symbol} Stock Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Close Price (in currency)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print("\nKey KPIs:")
    for key, value in kpis.items():
        print(f"{key}: {value}")
    
    print("\nCompany Summary:")
    print(summary)

def main():
    """
    Main function to interact with the user.
    """
    ticker_symbol = input("Enter the ticker symbol (e.g., MSFT): ").upper()
    period = input("Enter the period (e.g., 1d, 5d, 1mo, 1y, etc.): ")
    
    data, summary, kpis = fetch_data(ticker_symbol, period)
    plot_data(data, ticker_symbol, summary, kpis)
    
    # Provide a link to the Yahoo Finance page for the given ticker
    print("\nSource:")
    print(f"https://finance.yahoo.com/quote/{ticker_symbol}")

if __name__ == "__main__":
    main()


# In[6]:


import streamlit as st


# In[10]:


def fetch_data(ticker_symbol, period="1y"):
    """
    Fetch historical data and KPIs for the given ticker symbol and period.
    """
    ticker = yf.Ticker(ticker_symbol)
    data = ticker.history(period=period)
    summary = ticker.info.get('longBusinessSummary', 'No summary available.')
    
    kpis = {
        "Market Cap": ticker.info.get("marketCap"),
        "P/E Ratio": ticker.info.get("trailingPE"),
        "Dividend Yield": ticker.info.get("dividendYield"),
        "52 Week High": ticker.info.get("fiftyTwoWeekHigh"),
        "52 Week Low": ticker.info.get("fiftyTwoWeekLow"),
        "Beta": ticker.info.get("beta"),
        "Volume": ticker.info.get("volume"),
        "Earnings Per Share (EPS)": ticker.info.get("trailingEps"),
        "Price-to-Book Ratio (P/B)": ticker.info.get("priceToBook"),
        "Return on Equity (ROE)": ticker.info.get("returnOnEquity"),
        "Debt-to-Equity Ratio": ticker.info.get("debtToEquity"),
        "Current Ratio": ticker.info.get("currentRatio"),
        "Operating Margin": ticker.info.get("operatingMargin")
    }
    
    return data, summary, kpis

def plot_data(data, ticker_symbol, summary, kpis):
    """
    Plot the historical data, display KPIs, and the company's summary.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label='Close Price', color='blue')
    plt.title(f'{ticker_symbol} Stock Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Close Price (in currency)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print("\nKey KPIs:")
    for key, value in kpis.items():
        print(f"{key}: {value}")
    
    print("\nCompany Summary:")
    print(summary)

def main():
    st.title("Interactive Stock Dashboard")
    
    ticker_symbol = st.text_input("Enter the ticker symbol (e.g., MSFT):", "MSFT").upper()
    period = st.selectbox("Select the period:", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"])
    
    if st.button("Fetch Data"):
        data, summary, kpis = fetch_data(ticker_symbol, period)
        plot_data(data, ticker_symbol, summary, kpis)
        
        st.write("\nSource:")
        st.write(f"https://finance.yahoo.com/quote/{ticker_symbol}")

if __name__ == "__main__":
    main()


# In[8]:


from ipykernel import kernelapp as app

