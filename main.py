import numpy as np
import powerlaw
import pandas as pd
import streamlit as st
import plotly.express as px
from streamlit_extras.dataframe_explorer import dataframe_explorer

st.set_page_config(layout='wide')

if 'cpi' not in st.session_state:
    st.session_state['cpi'] = {1913 + i: 304.3 / v for i, v in enumerate(
        [9.9, 10.0, 10.1, 10.9, 12.8, 15.0, 17.3, 20.0, 17.9, 16.8, 17.1, 17.1, 17.5, 17.7, 17.4, 17.2, 17.2, 16.7,
         15.2, 13.6, 12.9, 13.4, 13.7, 13.9, 14.4, 14.1, 13.9, 14.0, 14.7, 16.3, 17.3, 17.6, 18.0, 19.5, 22.3, 24.0,
         23.8, 24.1, 26.0, 26.6, 26.8, 26.9, 26.8, 27.2, 28.1, 28.9, 29.2, 29.6, 29.9, 30.3, 30.6, 31.0, 31.5, 32.5,
         33.4, 34.8, 36.7, 38.8, 40.5, 41.8, 44.4, 49.3, 53.8, 56.9, 60.6, 65.2, 72.6, 82.4, 90.9, 96.5, 99.6, 103.9,
         107.6, 109.6, 113.6, 118.3, 124.0, 130.7, 136.2, 140.3, 144.5, 148.2, 152.4, 156.9, 160.5, 163.0, 166.6, 172.2,
         177.1, 179.9, 184.0, 188.9, 195.3, 201.6, 207.3, 215.3, 214.5, 218.1, 224.9, 229.6, 233.0, 236.7, 237.0, 240.0,
         245.1, 251.1, 255.7, 258.8, 271.0, 292.7, 304.3]
    )}


def preprocess(df, column, groupby='', top_n=-1):
    df[column] = pd.to_numeric(df[column], errors='coerce')
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


def get_hist(x):
    log_bins = np.logspace(np.log10(x.min()), np.log10(x.max()), 50)
    hist, bin_edges = np.histogram(x, bins=log_bins, density=True)
    return pd.Series({'hist': hist, 'bin_edges': bin_edges[:-1]})


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
        cpi_column = st.selectbox('CPI Column', [''] + list(df.columns))
        if cpi_column != '' and column != '':
            df[column] *= df[cpi_column].map(st.session_state['cpi'])
        year2decade = st.checkbox('Year to Decade')
        if year2decade and groupby != '':
            df = df.dropna(subset=[groupby])
            df[groupby] = df[groupby].astype(int) // 10 * 10
        log_rank_size = st.checkbox('Log-Rank vs. Log-Size', value=True)
    placeholder.dataframe(df, use_container_width=True)

    if column == '':
        st.stop()

with right:
    data = preprocess(df, column, groupby, top_n)

    kwargs = dict(data_frame=data, x=column, y='Rank', color=groupby, log_x=log_rank_size, log_y=log_rank_size, opacity=0.5, template='ggplot2')
    fig = px.scatter(**kwargs)

    data['Rank'] = data['Rank'] - 0.5
    if pl_fit:
        mask = data.groupby(groupby)[column].transform(
            lambda x: x >= powerlaw.Fit(x, verbose=False).power_law.xmin)
        data_2nd = data[~mask]
        data = data[mask]
    kwargs.update(dict(data_frame=data, trendline='ols', trendline_options=dict(log_x=True, log_y=True)))
    ols_fig = px.scatter(**kwargs)
    for trace in ols_fig.data:
        if trace.mode == 'lines':
            trace.line.dash = 'dot'
            fig.add_trace(trace)
    if pl_fit:
        kwargs['data_frame'] = data_2nd
        ols_fig_2nd = px.scatter(**kwargs)
        for trace in ols_fig_2nd.data:
            if trace.mode == 'lines':
                trace.line.dash = 'dot'
                fig.add_trace(trace)

    tabs = st.tabs(['Rank vs. Size', 'PDF', 'HHI'])
    with tabs[0]:
        st.plotly_chart(fig)
        # fig.write_json('tmp/rank_size.json')
    with tabs[1]:
        pdf = data.groupby(groupby)[column].apply(get_hist).reset_index()
        pdf = pdf.pivot(index=groupby, columns='level_1', values=column).reset_index().explode(['hist', 'bin_edges'])
        pdf = pdf[pdf['hist'] > 0]
        pdf_fig = px.scatter(pdf, x='bin_edges', y='hist', color=groupby, log_x=True, log_y=True, trendline='ols',
                             trendline_options=dict(log_x=True, log_y=True),
                             labels={'hist': 'Probability', 'bin_edges': column})
        for trace in pdf_fig.data:
            if trace.mode == 'lines':
                trace.line.dash = 'dot'
        st.plotly_chart(pdf_fig)
        # pdf_fig.write_json('tmp/pdf.json')
        st.caption('Note: 50 bins for each group')
        if pl_fit:
            pdf_2nd = data_2nd.groupby(groupby)[column].apply(get_hist).reset_index()
            pdf_2nd = pdf_2nd.pivot(index=groupby, columns='level_1', values=column).reset_index().explode(
                ['hist', 'bin_edges'])
            pdf_2nd = pdf_2nd[pdf_2nd['hist'] > 0]
            pdf_fig_2nd = px.scatter(pdf_2nd, x='bin_edges', y='hist', color=groupby, log_x=True, log_y=True,
                                     trendline='ols', trendline_options=dict(log_x=True, log_y=True),
                                     labels={'hist': 'Probability', 'bin_edges': column})
            for trace in pdf_fig_2nd.data:
                if trace.mode == 'lines':
                    trace.line.dash = 'dot'
            st.plotly_chart(pdf_fig_2nd)
            st.caption('Note: 2nd fit is only for the data above the xmin of the 1st fit')
    with tabs[2]:
        hhi = data.groupby(groupby)[column].apply(lambda x: ((x / x.sum())**2).sum() * 10000).reset_index()
        hhi_fig = px.scatter(hhi, x=groupby, y=column)
        st.plotly_chart(hhi_fig)
        # hhi.to_csv('tmp/hhi.csv', index=False)

    ols_results = px.get_trendline_results(ols_fig)
    if groupby == '':
        ols_results[groupby] = column
    alpha_gab = ols_results.set_index(groupby)['px_fit_results'].apply(
        lambda x: pd.Series([-x.params[1] + 1, x.bse[1] * 1.96, 'Gabaix'])).reset_index()
    results = data.groupby(groupby)[column].apply(lambda x: powerlaw.Fit(x, xmin=x.min()))
    r_pl = results.apply(lambda x: x.distribution_compare('power_law', 'lognormal')[0]).reset_index()
    alpha_pl = results.apply(lambda x: pd.Series([x.power_law.alpha, x.power_law.sigma * 1.96, 'Clauset'])).reset_index()
    alpha_df = pd.concat([alpha_gab, alpha_pl]).rename(columns={0: 'alpha', 1: 'error', 2: 'model'})
    alpha_df.to_csv('alpha.csv', index=False)
    tabs = st.tabs(['Alpha', 'R', 'Obs'])
    with tabs[0]:
        fig = px.scatter(alpha_df, x=groupby, y='alpha', error_y='error', color='model')
        st.plotly_chart(fig, use_container_width=True)
        if pl_fit:
            st.caption('Note: 2nd fit is only for the data above the xmin of the 1st fit')
            ols_results_2nd = px.get_trendline_results(ols_fig_2nd)
            if groupby == '':
                ols_results_2nd[groupby] = column
            alpha_gab_2nd = ols_results_2nd.set_index(groupby)['px_fit_results'].apply(
                lambda x: pd.Series([-x.params[1] + 1, x.bse[1] * 1.96, 'Gabaix'])).reset_index()
            results_2nd = data_2nd.groupby(groupby)[column].apply(lambda x: powerlaw.Fit(x, xmin=x.min()))
            alpha_pl_2nd = results_2nd.apply(
                lambda x: pd.Series([x.power_law.alpha, x.power_law.sigma * 1.96, 'Clauset'])).reset_index()
            alpha_df_2nd = pd.concat([alpha_gab_2nd, alpha_pl_2nd]).rename(columns={0: 'alpha', 1: 'error', 2: 'model'})
            fig = px.scatter(alpha_df_2nd, x=groupby, y='alpha', error_y='error', color='model')
            st.plotly_chart(fig, use_container_width=True)
    with tabs[1]:
        fig = px.scatter(r_pl, x=groupby, y=column, labels={column: 'R'})
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            'R: Loglikelihood ratio of the two distributions’ fit to the data. If greater than 0, the power_law distribution is preferred. If less than 0, the lognormal distribution is preferred.')
    with tabs[2]:
        fig = px.scatter(data.groupby(groupby).size().reset_index(), x=groupby, y=0, labels={'0': 'Number of Obs'})
        st.plotly_chart(fig, use_container_width=True)
        if pl_fit:
            st.caption('Note: 2nd fit is only for the data above the xmin of the 1st fit')
            fig = px.scatter(data_2nd.groupby(groupby).size().reset_index(), x=groupby, y=0,
                             labels={'0': 'Number of Obs'})
            st.plotly_chart(fig, use_container_width=True)
