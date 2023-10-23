#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import yfinance as yf
import plotly.graph_objects as go


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
    Plot the historical data, display KPIs, and the company's summary using Plotly.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
    fig.update_layout(title=f'{ticker_symbol} Stock Price Over Time', xaxis_title='Date', yaxis_title='Close Price (in currency)')
    
    st.plotly_chart(fig)  # Display the chart in Streamlit
    
    st.subheader("Key KPIs:")
    for key, value in kpis.items():
        st.write(f"{key}: {value}")
    
    st.subheader("Company Summary:")
    st.write(summary)

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


# In[ ]:




