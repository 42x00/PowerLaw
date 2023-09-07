import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout='wide')

left, right = st.columns(2)

with left:
    uploaded_file = st.file_uploader("File")
    if uploaded_file is None:
        st.stop()
    df = pd.read_csv(uploaded_file)
    st.dataframe(df, use_container_width=True)

    cols = list(df.columns)
    x = st.selectbox('x', cols, index=0)
    y = st.selectbox('y', cols, index=cols.index('alpha'))
    groupby = st.selectbox('group', cols, index=cols.index('group'))
    model = st.selectbox('model', ['pl', 'gab'])

with right:
    data = df[df['model'] == model]
    fig = px.line(data, x=x, y=y, color=groupby, markers=True)
    st.plotly_chart(fig, use_container_width=True)
