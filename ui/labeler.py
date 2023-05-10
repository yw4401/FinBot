import numpy as np
import streamlit as st
import pandas as pd


TEXT_COL = "chunk"
LABEL_COL = "relevant"


LABEL_DICT = {
    "Relevant": 0,
    "Not Relevant": 1,
    "Not Labeled": 2
}


source_df = None
with st.sidebar:
    uploaded_file = st.file_uploader("Upload dataframe to label", type="parquet")

if uploaded_file:
    source_df = pd.read_parquet(uploaded_file)
    if LABEL_COL not in source_df:
        source_df[LABEL_COL] = "Not Labeled"
    if TEXT_COL not in source_df.columns:
        st.write(":red[Please ensure that the df has column %s]" % TEXT_COL)
    else:
        st.write("## Label the following chunks as relevant or not")

        for i in range(len(source_df.index)):
            st.divider()
            st.write(source_df[TEXT_COL].iloc[i])
            current_val = source_df.iloc[i, source_df.columns.get_loc(LABEL_COL)]
            val = 2
            if current_val in LABEL_DICT:
                val = LABEL_DICT[current_val]
            label = st.radio("Is this chunk relevant?", ["Relevant", "Not Relevant", "Not Labeled"],
                             key=str(i), index=val)
            if label:
                source_df.iloc[i, source_df.columns.get_loc(LABEL_COL)] = label

with st.sidebar:
    if uploaded_file and source_df is not None:
        st.download_button(
            label="Download labeled data",
            data=source_df.to_parquet(index=False),
            file_name='labeled.parquet',
            mime='application/octet-stream',
        )
