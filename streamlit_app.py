import powerlaw
import pandas as pd
import streamlit as st
import plotly.express as px
from streamlit_extras.dataframe_explorer import dataframe_explorer

st.set_page_config(layout='wide')


def preprocess(df, column, groupby='', top_n=-1):
    if groupby == '':
        df[groupby] = column
    df = df[df[column] > 0][[groupby, column]]
    if top_n > 0:
        data = df.groupby(groupby)[column].nlargest(top_n).droplevel(1).reset_index()
    else:
        data = df.sort_values(by=[groupby, column], ascending=[True, False])
    data['Rank'] = data.groupby(groupby).cumcount() + 1
    data[groupby] = data[groupby].astype(str)
    return data


left, right = st.columns(2)

with left:
    uploaded_file = st.file_uploader("File")
    if uploaded_file is None:
        st.stop()
    df = pd.read_csv(uploaded_file)

    placeholder = st.empty()

    column = st.selectbox('Column', [''] + list(df.columns))

    with st.expander('Options'):
        groupby = st.selectbox('Group by', [''] + list(df.columns))
        top_n = st.number_input('Top N', value=10000)
        pl_fit = st.checkbox('PowerLaw Fit')
        df = dataframe_explorer(df)
    placeholder.dataframe(df.head(1000), use_container_width=True)

    if column == '':
        st.stop()

with right:
    data = preprocess(df, column, groupby, top_n)

    kwargs = dict(data_frame=data, x=column, y='Rank', color=groupby, log_x=True, log_y=True, opacity=0.5)
    fig = px.scatter(**kwargs)

    data['Rank'] = data['Rank'] - 0.5
    if pl_fit:
        mask = data.groupby(groupby)[column].transform(
            lambda x: x >= powerlaw.Fit(x, discrete=True, verbose=False).power_law.xmin)
        data = data[mask]
    kwargs.update(dict(data_frame=data, trendline='ols', trendline_options=dict(log_x=True, log_y=True)))
    ols_fig = px.scatter(**kwargs)
    for trace in ols_fig.data:
        if trace.mode == 'lines':
            trace.line.dash = 'dot'
            fig.add_trace(trace)

    st.plotly_chart(fig)

    ols_results = px.get_trendline_results(ols_fig)
    if groupby == '':
        ols_results[groupby] = column
    alpha_gab = ols_results.set_index(groupby)['px_fit_results'].apply(
        lambda x: pd.Series([-x.params[1] + 1, x.bse[1], 'gab'])).reset_index()
    alpha_pl = data.groupby(groupby)[column].apply(lambda x: powerlaw.Fit(x, xmin=x.min(), discrete=True)).apply(
        lambda x: pd.Series(
            [x.power_law.alpha, x.distribution_compare('power_law', 'lognormal')[1], 'pl'])).reset_index()
    alpha_df = pd.concat([alpha_gab, alpha_pl]).rename(columns={0: 'alpha', 1: 'error', 2: 'model'})
    fig = px.scatter(alpha_df, x=groupby, y='alpha', error_y='error', color='model')
    st.plotly_chart(fig, use_container_width=True)
