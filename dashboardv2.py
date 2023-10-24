#!/usr/bin/env python
# coding: utf-8

# In[5]:


import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
#import summarizer.ner as ner

def fetch_data(ticker_symbol, period="1y"):
    
    ticker = yf.Ticker(ticker_symbol)
    data = ticker.history(period=period)
    summary = ticker.info.get('longBusinessSummary', 'No summary available.')

    kpis = {
    "Basic": {
        "Market Cap($)": ticker.info.get("marketCap"),
        "Total Enterprise Value (TEV)($)": ticker.info.get("enterpriseValue"),
        "Total Revenues($)": ticker.info.get("totalRevenue")
    },
    "Profitability": {
        "Gross Profit Margin(%)": ticker.info.get("grossMargins"),
        "EBITDA Margin(%)": ticker.info.get("ebitdaMargins"),
        "Operating Margin(%)": ticker.info.get("operatingMargin"),
        "Net Profit Margin(%)": ticker.info.get("netProfitMargin"),
        "Pre-Tax Profit Margin(%)": ticker.info.get("profitMargins")
    },
    "Per Share": {
        "Revenue per Share($)": ticker.info.get("revenuePerShare"),
        "EPS Diluted($)": ticker.info.get("trailingEps")
    },
    "Employees": {
        "Total Employees": ticker.info.get("fullTimeEmployees")
    },
    "Valuation": {
        "EV/Sales": ticker.info.get("enterpriseToRevenue"),
        "P/E": ticker.info.get("trailingPE"),
        "EV/EBITDA": ticker.info.get("enterpriseToEbitda"),
        "P/B": ticker.info.get("priceToBook"),
        
    },
    "Forward Valuation": {
        
        "Forward P/E": ticker.info.get("forwardPE")
    }
}

    return data, summary, kpis

def plot_data(data, ticker_symbol, summary, kpis):

    st.markdown("### **Key KPIs:**")
    
    for category, metrics in kpis.items():
        
        # Using Streamlit expander to create collapsible sections for each category
        with st.expander(category, expanded=False):
            # Begin the table markdown string
            table_md = "| Metric | Value |\n|---|---|\n"
            
            for key, value in metrics.items():
                # Append each KPI to the table markdown string
                table_md += f"| {key} | {value if value else 'N/A'} |\n"
            
            # Display the markdown table
            st.markdown(table_md)

    st.markdown("### **Company Summary:**")
    st.write(summary)

    st.write("\nSource:")
    st.markdown(f"[Yahoo Finance](https://finance.yahoo.com/quote/{ticker_symbol})")

def main():
    st.title("Response")
    
    ticker_symbol = st.text_input("Enter the ticker symbol (e.g., MSFT):", "MSFT").upper()
    period = st.selectbox("Select the period:", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"])
    
    if st.button("Fetch Data"):
        data, summary, kpis = fetch_data(ticker_symbol, period)
        plot_data(data, ticker_symbol, summary, kpis)
        
        st.write(f"https://finance.yahoo.com/quote/{ticker_symbol}")

if __name__ == "__main__":
    main()
    


# In[ ]:




