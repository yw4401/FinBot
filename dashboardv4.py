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

    # Display the QA if exists
    resp_text = []
    answer = response["qa"]["answer"]
    if "sorry" not in answer.lower():
        st.write("### Direct Answer")
        resp_text.append(answer)
        st.write(escape_markdown(ui.tex_escape(answer)))
        with st.expander("Sources"):
            format_sources(response["qa"]["sources"])
            st.write("")
        st.write("")  # Add an empty line for separation

    # Loop through each summary and display its title and keypoints
    valid_summaries = [summary for summary in response["summaries"] if "impossible to answer" not in summary["title"].lower()]
    if len(valid_summaries) > 0:
        st.write("### Related to Your Query")
    for summary in valid_summaries:
        title = ui.tex_escape(escape_markdown(summary['title'].strip()))
        st.write(f"**{ui.tex_escape(escape_markdown(summary['title'].strip()))}**\n")
        resp_text.append(title)
        # Create bullet points for each keypoint
        for keypoint in summary["keypoints"]:
            resp_text.append(keypoint)
            st.write(f"- {ui.tex_escape(escape_markdown(keypoint))}\n")

        st.write("")
        with st.expander("Sources"):
            format_sources(summary["sources"])
            st.write("")
        st.write("")  # Add an empty line for separation

    if len(resp_text) == 0:
        st.write("Sorry, we were unable to find anything related to your query.")

    return resp_text


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


def create_screener(user_text, resp_text, period):
    str_resp = ". ".join(resp_text)
    with st.spinner("Extracting symbols"):
        ticker_symbol = ner.extract_company_ticker(user_text, str_resp)
    if len(ticker_symbol) != 0:
        with st.spinner("Fetching Info"):
            results = asyncio.run(ui.fetch_all_tickers(ticker_symbol, user_text, period))
            if len(results) > 0:
                st.write("### You may be interested in:")
            for ticker, data, summary, result in results[:3]:
                with st.expander(label=ticker, expanded=False):
                    plot_data(data, ticker, summary, result)
                    st.write(f"https://finance.yahoo.com/quote/{ticker}")


def qa_ux(user_text, period):
    qa_resp = display_summary_response(user_text, period)
    if len(qa_resp) > 0:
        create_screener(user_text, qa_resp, period)


def investment_advice():
    return False


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
    st.write("**Note: As a large language model, FinBot can only guarantee best effort correctness of the response**")
    st.write("_Always double check the responses for substantial decisions_")

    # Follow up logic

    if st.button("Ask Finbot") and len(user_text.strip()) > 0:
        if investment_advice():
            investment_ux(user_text, period)
        else:
            qa_ux(user_text, period)


if __name__ == "__main__":
    main()
