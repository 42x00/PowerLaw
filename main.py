import numpy as np
import powerlaw
import pandas as pd
import streamlit as st
import plotly.express as px
from streamlit_extras.dataframe_explorer import dataframe_explorer

st.set_page_config(layout='wide')

if 'cpi' not in st.session_state:
    st.session_state['cpi'] = \
        {1913: 9.9, 1914: 10.0, 1915: 10.1, 1916: 10.9, 1917: 12.8, 1918: 15.0, 1919: 17.3, 1920: 20.0, 1921: 17.9,
         1922: 16.8, 1923: 17.1, 1924: 17.1, 1925: 17.5, 1926: 17.7, 1927: 17.4, 1928: 17.2, 1929: 17.2, 1930: 16.7,
         1931: 15.2, 1932: 13.6, 1933: 12.9, 1934: 13.4, 1935: 13.7, 1936: 13.9, 1937: 14.4, 1938: 14.1, 1939: 13.9,
         1940: 14.0, 1941: 14.7, 1942: 16.3, 1943: 17.3, 1944: 17.6, 1945: 18.0, 1946: 19.5, 1947: 22.3, 1948: 24.0,
         1949: 23.8, 1950: 24.1, 1951: 26.0, 1952: 26.6, 1953: 26.8, 1954: 26.9, 1955: 26.8, 1956: 27.2, 1957: 28.1,
         1958: 28.9, 1959: 29.2, 1960: 29.6, 1961: 29.9, 1962: 30.3, 1963: 30.6, 1964: 31.0, 1965: 31.5, 1966: 32.5,
         1967: 33.4, 1968: 34.8, 1969: 36.7, 1970: 38.8, 1971: 40.5, 1972: 41.8, 1973: 44.4, 1974: 49.3, 1975: 53.8,
         1976: 56.9, 1977: 60.6, 1978: 65.2, 1979: 72.6, 1980: 82.4, 1981: 90.9, 1982: 96.5, 1983: 99.6, 1984: 103.9,
         1985: 107.6, 1986: 109.6, 1987: 113.6, 1988: 118.3, 1989: 124.0, 1990: 130.7, 1991: 136.2, 1992: 140.3,
         1993: 144.5, 1994: 148.2, 1995: 152.4, 1996: 156.9, 1997: 160.5, 1998: 163.0, 1999: 166.6, 2000: 172.2,
         2001: 177.1, 2002: 179.9, 2003: 184.0, 2004: 188.9, 2005: 195.3, 2006: 201.6, 2007: 207.3, 2008: 215.3,
         2009: 214.5, 2010: 218.1, 2011: 224.9, 2012: 229.6, 2013: 233.0, 2014: 236.7, 2015: 237.0, 2016: 240.0,
         2017: 245.1, 2018: 251.1, 2019: 255.7, 2020: 258.8, 2021: 271.0, 2022: 292.7, 2023: 304.3}


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
        if cpi_column != '':
            df[column] /= df[cpi_column].map(st.session_state['cpi'])
        year2decade = st.checkbox('Year to Decade')
        if year2decade:
            df = df.dropna(subset=[groupby])
            df[groupby] = df[groupby].astype(int) // 10 * 10
    placeholder.dataframe(df, use_container_width=True)

    if column == '':
        st.stop()

with right:
    data = preprocess(df, column, groupby, top_n)

    kwargs = dict(data_frame=data, x=column, y='Rank', color=groupby, log_x=True, log_y=True, opacity=0.5)
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

    tabs = st.tabs(['Log-Rank vs. Log-Size', 'CDF'])
    with tabs[0]:
        st.plotly_chart(fig)
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
        st.caption('Note: 50 bins for each group')

    ols_results = px.get_trendline_results(ols_fig)
    if groupby == '':
        ols_results[groupby] = column
    alpha_gab = ols_results.set_index(groupby)['px_fit_results'].apply(
        lambda x: pd.Series([-x.params[1] + 1, x.bse[1] * 1.96, 'gab'])).reset_index()
    results = data.groupby(groupby)[column].apply(lambda x: powerlaw.Fit(x, xmin=x.min()))
    r_pl = results.apply(lambda x: x.distribution_compare('power_law', 'lognormal')[0]).reset_index()
    alpha_pl = results.apply(lambda x: pd.Series([x.power_law.alpha, x.power_law.sigma * 1.96, 'pl'])).reset_index()
    alpha_df = pd.concat([alpha_gab, alpha_pl]).rename(columns={0: 'alpha', 1: 'error', 2: 'model'})
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
                lambda x: pd.Series([-x.params[1] + 1, x.bse[1] * 1.96, 'gab'])).reset_index()
            results_2nd = data_2nd.groupby(groupby)[column].apply(lambda x: powerlaw.Fit(x, xmin=x.min()))
            alpha_pl_2nd = results_2nd.apply(
                lambda x: pd.Series([x.power_law.alpha, x.power_law.sigma * 1.96, 'pl'])).reset_index()
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
