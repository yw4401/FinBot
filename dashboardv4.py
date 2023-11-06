import asyncio

import plotly.graph_objects as go
import streamlit as st
from streamlit.string_util import escape_markdown

import summarizer.ner as ner
import summarizer.uiinterface as ui


def format_sources(sources):
    existing_sources = set()
    existing_title = set()
    for doc in sources:
        if doc.metadata['url'] not in existing_sources and doc.metadata["title"] not in existing_title:
            st.write(f"- [{doc.metadata['title']}]({doc.metadata['url']})")
            existing_sources.add(doc.metadata['url'])
            existing_title.add(doc.metadata["title"])


def display_summary_response(text, period):
    with st.spinner("Thinking"):
        response = ui.finbot_response(text, period)

    # Display the QA placeholder
    st.write("### Response")
    st.write(escape_markdown(ui.tex_escape(response["qa"]["answer"])))
    with st.expander("Sources"):
        format_sources(response["qa"]["sources"])
        st.write("")
    st.write("")  # Add an empty line for separation

    # Loop through each summary and display its title and keypoints
    for summary in response["summaries"]:
        st.write(f"**{ui.tex_escape(escape_markdown(summary['title'].strip()))}**\n")
        # Create bullet points for each keypoint
        for keypoint in summary["keypoints"]:
            st.write(f"- {ui.tex_escape(escape_markdown(keypoint))}\n")

        st.write("")
        with st.expander("Sources"):
            format_sources(summary["sources"])
            st.write("")
        st.write("")  # Add an empty line for separation

    return response


def plot_data(data, ticker_symbol, summary, kpis):
    """
    Plot the historical data, display KPIs, and the company's summary using Plotly.
    """
    if data is not None:
        fig = go.Figure(data=[go.Candlestick(x=data.index,
                                             open=data['open'],
                                             high=data['high'],
                                             low=data['low'],
                                             close=data['close'])])
        fig.update_layout(title=f'{ticker_symbol} Stock Price Over Time', xaxis_title='Date',
                          yaxis_title='Price (dollars)')
        st.plotly_chart(fig, use_container_width=True)

    for category, metrics in kpis.items():
        if len(metrics) == 0:
            continue

        # Using Streamlit expander to create collapsible sections for each category
        st.write(f"##### {category}")
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


def create_screener(user_text, qa_resp, period):
    str_resp = ner.format_response_for_ner(qa_resp)
    with st.spinner("Extracting symbols"):
        ticker_symbol = ner.extract_company_ticker(user_text, str_resp)
        print(ticker_symbol)
    if len(ticker_symbol) != 0:
        st.write("### Screener")
        with st.spinner("Fetching Info"):
            results = asyncio.run(ui.fetch_all_tickers(ticker_symbol, user_text, period))
            for ticker, data, summary, result in results[:3]:
                with st.expander(label=ticker, expanded=False):
                    plot_data(data, ticker, summary, result)
                    st.write(f"https://finance.yahoo.com/quote/{ticker}")


def qa_ux(user_text, period):
    qa_resp = display_summary_response(user_text, period)
    create_screener(user_text, qa_resp, period)


def investment_advice():
    return True


def investment_ux(user_text, period):
    # add extra ui
    pass
    # add output
    pass
    pass


def main():
    st.title("FinBot")

    user_text = st.text_input("Enter your question here:")
    period = st.selectbox("Select the period:",
                          ["1d", "5d", "1mo", "3mo", "6mo"])

    # Follow up logic

    if st.button("Ask Finbot") and len(user_text.strip()) > 0:
        if investment_advice():
            investment_ux(user_text, period)
        else:
            qa_ux(user_text, period)


if __name__ == "__main__":
    main()
