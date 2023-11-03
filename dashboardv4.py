import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import summarizer.ner as ner
import summarizer.uiinterface as ui


def fetch_data_alt(ticker_symbol, period="1y"):
    ticker = ui.YFinance(ticker_symbol)
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

    return None, summary, kpis


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


def display_summary_response(text, period):
    response = ui.finbot_response(text, period)

    # Display the QA placeholder
    st.write(response["qa"])
    st.write("")  # Add an empty line for separation

    # Loop through each summary and display its title and keypoints
    st.write("### You may be interested in:")
    for summary in response["summaries"]:
        # Create bullet points for each keypoint
        for keypoint in summary["keypoints"]:
            st.markdown(f"- {keypoint}")

        st.write("")  # Add an empty line for separation

    return response


def plot_data(data, ticker_symbol, summary, kpis):
    """
    Plot the historical data, display KPIs, and the company's summary using Plotly.
    """
    if data:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
        fig.update_layout(title=f'{ticker_symbol} Stock Price Over Time', xaxis_title='Date',
                          yaxis_title='Close Price (in currency)')

        st.plotly_chart(fig)

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
    st.title("FinBot")

    user_text = st.text_input("Enter your question here:")

    #   ticker_symbol = st.text_input("Enter the ticker symbol (e.g., MSFT):", "MSFT").upper()
    period = st.selectbox("Select the period:",
                          ["1d", "5d", "1mo", "3mo", "6mo"])

    if st.button("Ask Finbot") and len(user_text.strip()) > 0:
        qa_resp = display_summary_response(user_text, period)
        ticker_symbol = ner.extract_company_ticker(user_text, qa_resp)
        data, summary, kpis = fetch_data_alt(ticker_symbol, period)

        plot_data(data, ticker_symbol, summary, kpis)

        st.write(f"https://finance.yahoo.com/quote/{ticker_symbol}")


if __name__ == "__main__":
    main()
