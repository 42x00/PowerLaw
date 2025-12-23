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

def get_ccdf(x):
    x = pd.to_numeric(x, errors='coerce').dropna().to_numpy()
    x = x[x > 0]
    if x.size == 0:
        return pd.Series({'ccdf': np.array([]), 'bin_edges': np.array([])})

    log_bins = np.logspace(np.log10(x.min()), np.log10(x.max()), 50)
    thresholds = log_bins[:-1]  # match your PDF x-axis convention

    xs = np.sort(x)
    n = xs.size
    idx = np.searchsorted(xs, thresholds, side='left')
    ccdf = (n - idx) / n  # P(X >= threshold)

    return pd.Series({'ccdf': ccdf, 'bin_edges': thresholds})


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
    # Pre-calc data
    data = preprocess(df, column, groupby, top_n)

    # Prepare figure placeholders for later downloads
    rank_fig = None
    pdf_fig = None
    pdf_fig_2nd = None
    hhi_fig = None
    alpha_fig = None
    alpha_fig_2nd = None
    r_fig = None
    obs_fig = None
    obs_fig_2nd = None

    # --- Rank vs Size (with OLS / power law overlays) ---
    kwargs = dict(
        data_frame=data,
        x=column,
        y='Rank',
        color=groupby,
        log_x=log_rank_size,
        log_y=log_rank_size,
        opacity=0.5,
        template='ggplot2'
    )
    rank_fig = px.scatter(**kwargs)

    data['Rank'] = data['Rank'] - 0.5

    data_2nd = None
    if pl_fit:
        mask = data.groupby(groupby)[column].transform(
            lambda x: x >= powerlaw.Fit(x, verbose=False).power_law.xmin
        )
        data_2nd = data[~mask]
        data = data[mask]

    kwargs.update(dict(
        data_frame=data,
        trendline='ols',
        trendline_options=dict(log_x=True, log_y=True)
    ))
    ols_fig = px.scatter(**kwargs)

    for trace in ols_fig.data:
        if trace.mode == 'lines':
            trace.line.dash = 'dot'
            rank_fig.add_trace(trace)

    if pl_fit and data_2nd is not None and not data_2nd.empty:
        kwargs['data_frame'] = data_2nd
        ols_fig_2nd = px.scatter(**kwargs)
        for trace in ols_fig_2nd.data:
            if trace.mode == 'lines':
                trace.line.dash = 'dot'
                rank_fig.add_trace(trace)

    tabs_main = st.tabs(['Rank vs. Size', 'PDF', 'CCDF', 'HHI'])
    with tabs_main[0]:
        st.plotly_chart(rank_fig)

    # --- PDF tab ---
    with tabs_main[1]:
        pdf = data.groupby(groupby)[column].apply(get_hist).reset_index()
        pdf = pdf.pivot(index=groupby, columns='level_1', values=column).reset_index().explode(['hist', 'bin_edges'])
        pdf = pdf[pdf['hist'] > 0]
        pdf_fig = px.scatter(
            pdf,
            x='bin_edges',
            y='hist',
            color=groupby,
            log_x=True,
            log_y=True,
            trendline='ols',
            trendline_options=dict(log_x=True, log_y=True),
            labels={'hist': 'Probability', 'bin_edges': column}
        )
        for trace in pdf_fig.data:
            if trace.mode == 'lines':
                trace.line.dash = 'dot'
        st.plotly_chart(pdf_fig)
        st.caption('Note: 50 bins for each group')

        if pl_fit and data_2nd is not None and not data_2nd.empty:
            pdf_2nd = data_2nd.groupby(groupby)[column].apply(get_hist).reset_index()
            pdf_2nd = pdf_2nd.pivot(index=groupby, columns='level_1', values=column).reset_index().explode(
                ['hist', 'bin_edges'])
            pdf_2nd = pdf_2nd[pdf_2nd['hist'] > 0]
            pdf_fig_2nd = px.scatter(
                pdf_2nd,
                x='bin_edges',
                y='hist',
                color=groupby,
                log_x=True,
                log_y=True,
                trendline='ols',
                trendline_options=dict(log_x=True, log_y=True),
                labels={'hist': 'Probability', 'bin_edges': column}
            )
            for trace in pdf_fig_2nd.data:
                if trace.mode == 'lines':
                    trace.line.dash = 'dot'
            st.plotly_chart(pdf_fig_2nd)
            st.caption('Note: 2nd fit is only for the data above the xmin of the 1st fit')

    # --- CCDF tab ---
    with tabs_main[2]:
        ccdf = data.groupby(groupby)[column].apply(get_ccdf).reset_index()
        ccdf = ccdf.pivot(index=groupby, columns='level_1', values=column).reset_index() \
                .explode(['ccdf', 'bin_edges'])
        ccdf = ccdf[ccdf['ccdf'] > 0]

        ccdf_fig = px.scatter(
            ccdf,
            x='bin_edges',
            y='ccdf',
            color=groupby,
            log_x=True,
            log_y=True,
            trendline='ols',
            trendline_options=dict(log_x=True, log_y=True),
            labels={'ccdf': 'P(X ≥ x)', 'bin_edges': column}
        )
        for trace in ccdf_fig.data:
            if trace.mode == 'lines':
                trace.line.dash = 'dot'

        st.plotly_chart(ccdf_fig)
        st.caption('Note: 50 log-spaced thresholds per group; y is empirical P(X ≥ x).')

        if pl_fit and data_2nd is not None and not data_2nd.empty:
            ccdf_2nd = data_2nd.groupby(groupby)[column].apply(get_ccdf).reset_index()
            ccdf_2nd = ccdf_2nd.pivot(index=groupby, columns='level_1', values=column).reset_index() \
                            .explode(['ccdf', 'bin_edges'])
            ccdf_2nd = ccdf_2nd[ccdf_2nd['ccdf'] > 0]

            ccdf_fig_2nd = px.scatter(
                ccdf_2nd,
                x='bin_edges',
                y='ccdf',
                color=groupby,
                log_x=True,
                log_y=True,
                trendline='ols',
                trendline_options=dict(log_x=True, log_y=True),
                labels={'ccdf': 'P(X ≥ x)', 'bin_edges': column}
            )
            for trace in ccdf_fig_2nd.data:
                if trace.mode == 'lines':
                    trace.line.dash = 'dot'

            st.plotly_chart(ccdf_fig_2nd)
            st.caption('Note: 2nd fit is only for the data above the xmin of the 1st fit')

    # --- HHI tab ---
    with tabs_main[3]:
        hhi = data.groupby(groupby)[column].apply(lambda x: ((x / x.sum())**2).sum() * 10000).reset_index()
        hhi_fig = px.scatter(hhi, x=groupby, y=column)
        st.plotly_chart(hhi_fig)

    # --- Alpha, R, Obs calculations ---
    ols_results = px.get_trendline_results(ols_fig)
    if groupby == '':
        ols_results[groupby] = column

    alpha_gab = ols_results.set_index(groupby)['px_fit_results'].apply(
        lambda x: pd.Series([-x.params[1] + 1, x.bse[1] * 1.96, 'Gabaix'])
    ).reset_index()

    results = data.groupby(groupby)[column].apply(lambda x: powerlaw.Fit(x, xmin=x.min()))
    r_pl = results.apply(lambda x: x.distribution_compare('power_law', 'lognormal')[0]).reset_index()
    alpha_pl = results.apply(
        lambda x: pd.Series([x.power_law.alpha, x.power_law.sigma * 1.96, 'Clauset'])
    ).reset_index()

    alpha_df = pd.concat([alpha_gab, alpha_pl]).rename(columns={0: 'alpha', 1: 'error', 2: 'model'})
    alpha_df.to_csv('alpha.csv', index=False)

    tabs_stats = st.tabs(['Alpha', 'R', 'Obs'])

    # Alpha tab
    with tabs_stats[0]:
        alpha_fig = px.scatter(alpha_df, x=groupby, y='alpha', error_y='error', color='model')
        st.plotly_chart(alpha_fig, use_container_width=True)
        if pl_fit and data_2nd is not None and not data_2nd.empty:
            st.caption('Note: 2nd fit is only for the data above the xmin of the 1st fit')
            ols_results_2nd = px.get_trendline_results(ols_fig_2nd)
            if groupby == '':
                ols_results_2nd[groupby] = column
            alpha_gab_2nd = ols_results_2nd.set_index(groupby)['px_fit_results'].apply(
                lambda x: pd.Series([-x.params[1] + 1, x.bse[1] * 1.96, 'Gabaix'])
            ).reset_index()
            results_2nd = data_2nd.groupby(groupby)[column].apply(lambda x: powerlaw.Fit(x, xmin=x.min()))
            alpha_pl_2nd = results_2nd.apply(
                lambda x: pd.Series([x.power_law.alpha, x.power_law.sigma * 1.96, 'Clauset'])
            ).reset_index()
            alpha_df_2nd = pd.concat([alpha_gab_2nd, alpha_pl_2nd]).rename(
                columns={0: 'alpha', 1: 'error', 2: 'model'}
            )
            alpha_fig_2nd = px.scatter(alpha_df_2nd, x=groupby, y='alpha', error_y='error', color='model')
            st.plotly_chart(alpha_fig_2nd, use_container_width=True)

    # R tab
    with tabs_stats[1]:
        r_fig = px.scatter(r_pl, x=groupby, y=column, labels={column: 'R'})
        st.plotly_chart(r_fig, use_container_width=True)
        st.caption(
            'R: Loglikelihood ratio of the two distributions’ fit to the data. '
            'If greater than 0, the power_law distribution is preferred. '
            'If less than 0, the lognormal distribution is preferred.'
        )

    # Obs tab
    with tabs_stats[2]:
        obs_counts = data.groupby(groupby).size().reset_index()
        obs_fig = px.scatter(
            obs_counts,
            x=groupby,
            y=0,
            labels={'0': 'Number of Obs'}
        )
        st.plotly_chart(obs_fig, use_container_width=True)
        if pl_fit and data_2nd is not None and not data_2nd.empty:
            st.caption('Note: 2nd fit is only for the data above the xmin of the 1st fit')
            obs_counts_2nd = data_2nd.groupby(groupby).size().reset_index()
            obs_fig_2nd = px.scatter(
                obs_counts_2nd,
                x=groupby,
                y=0,
                labels={'0': 'Number of Obs'}
            )
            st.plotly_chart(obs_fig_2nd, use_container_width=True)

    # --- Download figures as JSON ---
    st.markdown("## Download Figures as JSON")

    def download_plotly_json(fig, label, filename):
        if fig is not None:
            st.download_button(
                label=label,
                data=fig.to_json(),
                file_name=filename,
                mime="application/json"
            )

    col_dl1, col_dl2, col_dl3 = st.columns(3)

    with col_dl1:
        download_plotly_json(rank_fig, "Download Rank vs Size", "rank_vs_size.json")
        download_plotly_json(pdf_fig, "Download PDF", "pdf.json")
        download_plotly_json(ccdf_fig, "Download CCDF", "ccdf.json")
        download_plotly_json(hhi_fig, "Download HHI", "hhi.json")

    with col_dl2:
        download_plotly_json(alpha_fig, "Download Alpha (1st fit)", "alpha.json")
        download_plotly_json(r_fig, "Download R (loglikelihood ratio)", "r.json")
        download_plotly_json(obs_fig, "Download Obs counts (1st fit)", "obs.json")

    with col_dl3:
        # Only meaningful when pl_fit is True and 2nd-fit figs exist
        download_plotly_json(pdf_fig_2nd, "Download PDF (2nd fit)", "pdf_2nd.json")
        download_plotly_json(alpha_fig_2nd, "Download Alpha (2nd fit)", "alpha_2nd.json")
        download_plotly_json(obs_fig_2nd, "Download Obs counts (2nd fit)", "obs_2nd.json")
