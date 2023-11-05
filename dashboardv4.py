import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from streamlit.string_util import escape_markdown

import summarizer.ner as ner
import summarizer.uiinterface as ui


def fetch_data_alt(ticker_symbol, query, response, period="1y"):
    ticker = ui.YFinance(ticker_symbol)
    summary = ticker.info.get('longBusinessSummary', 'No summary available.')
    market_cap = ticker.info.get("marketCap")
    if not market_cap:
        raise FileNotFoundError(ticker_symbol)

    kpis = ner.extract_relevant_field(query, summary, ticker)
    result = {}
    for g in kpis.groups:
        if g.group_title not in result:
            result[g.group_title] = {}
        for m in g.group_members:
            if m in ticker.info:
                result[g.group_title][m] = ticker.info[m]

    return None, summary, result


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
    with st.spinner("Thinking"):
        response = ui.finbot_response(text, period)

    # Display the QA placeholder
    st.write("### Response")
    st.write(escape_markdown(ui.tex_escape(response["qa"])))
    st.write("")  # Add an empty line for separation

    # Loop through each summary and display its title and keypoints
    for summary in response["summaries"]:
        st.write(f"###### {ui.tex_escape(escape_markdown(summary['title'].strip()))}\n")
        # Create bullet points for each keypoint
        for keypoint in summary["keypoints"]:
            st.write(f"- {ui.tex_escape(escape_markdown(keypoint))}\n")

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

    st.markdown(f"### **{ticker_symbol}**")

    for category, metrics in kpis.items():
        if len(metrics) == 0:
            continue

        # Using Streamlit expander to create collapsible sections for each category
        with st.expander(category, expanded=False):
            # Begin the table markdown string
            table_md = "| Metric | Value |\n|---|---|\n"

            for key, value in metrics.items():
                # Append each KPI to the table markdown string
                table_md += f"| {key} | {value if value else 'N/A'} |\n"

            # Display the markdown table
            st.write(table_md)
            st.write("\n")

    st.markdown("#### **Company Summary:**")
    st.write(summary)

    st.write("\nSource:")
    st.markdown(f"[Yahoo Finance](https://finance.yahoo.com/quote/{ticker_symbol})")


def main():
    st.title("FinBot")

    user_text = st.text_input("Enter your question here:")
    period = st.selectbox("Select the period:",
                          ["1d", "5d", "1mo", "3mo", "6mo"])

    if st.button("Ask Finbot") and len(user_text.strip()) > 0:
        qa_resp = display_summary_response(user_text, period)
        str_resp = ner.format_response_for_ner(qa_resp)
        with st.spinner("Extracting symbols"):
            ticker_symbol = ner.extract_company_ticker(user_text, str_resp)
            print(ticker_symbol)
        if len(ticker_symbol) != 0:
            for t in ticker_symbol:
                try:
                    with st.spinner("Fetching KPIs"):
                        data, summary, kpis = fetch_data_alt(t, user_text, str_resp, period)
                        plot_data(data, t, summary, kpis)
                        st.write(f"https://finance.yahoo.com/quote/{t}")
                    break
                except FileNotFoundError:
                    pass


if __name__ == "__main__":
    main()
