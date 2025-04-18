import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import powerlaw as pl

pio.templates.default = "ggplot2"

st.set_page_config(layout="wide")

def estimate_gabaix_alpha(sorted_data):
    return 1 - np.polyfit(np.log(sorted_data), np.log(np.arange(0.5, len(sorted_data) + 0.5)), 1)[0]

def estimate_clauset_alpha(sorted_data):
    return 1 + len(sorted_data) / np.sum(np.log(sorted_data / sorted_data[-1]))

left_column, right_column = st.columns(2)

with left_column:
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is None:
        st.stop()
    dataframe = pd.read_csv(uploaded_file)
    st.dataframe(dataframe, use_container_width=True)

    dataframe_columns = dataframe.columns.tolist()
    selected_column = st.selectbox("Select a column to analyze", [''] + dataframe_columns)
    if selected_column == '':
        st.stop()

    top_n = st.number_input("Top N values to consider", min_value=1, value=10000, step=1)

    data = pd.to_numeric(dataframe[selected_column], errors='coerce').dropna().values
    data = data[data > 0]
    data = np.sort(data)[::-1]
    data = data[:top_n]

with right_column:
    total = len(data)
    average = np.mean(data)
    minimum = np.min(data)
    maximum = np.max(data)

    rank = np.arange(1, total + 1)
    selected_column_label = ' '.join([word.capitalize() for word in selected_column.replace('_', ' ').split()])

    tab1, tab2, tab3, tab4 = st.tabs(["Rank Size", "Rank Size", "Log PDF", "Hill Plot"])

    with tab1:
        fig = px.scatter(x=data, y=rank, labels={'x': selected_column_label, 'y': 'Rank'})
        st.plotly_chart(fig, use_container_width=True, theme=None)

    with tab2:
        fig = px.scatter(x=data, y=rank, labels={'x': 'Log ' + selected_column_label, 'y': 'Log Rank'}, log_x=True, log_y=True, 
                         trendline='ols', trendline_options={'log_x': True, 'log_y': True}
                         )
        for trace in fig.data:
            if trace.mode == 'lines':
                trace.line.dash = 'dot'
        st.plotly_chart(fig, use_container_width=True, theme=None)

    with tab3:
        log_bins = np.logspace(np.log10(minimum), np.log10(maximum), num=50)
        hist, bin_edges = np.histogram(data, bins=log_bins, density=True)
        indices = hist > 0
        hist, bin_edges = hist[indices], bin_edges[:-1][indices]
        fig = px.scatter(x=bin_edges, y=hist, labels={'x': 'Log ' + selected_column_label, 'y': 'Log Probability Density'}, log_x=True, log_y=True, 
                         trendline='ols', trendline_options={'log_x': True, 'log_y': True}
                         )
        for trace in fig.data:
            if trace.mode == 'lines':
                trace.line.dash = 'dot'
        st.plotly_chart(fig, use_container_width=True, theme=None)

    with tab4:
        hill_plot_data = pd.DataFrame()
        hill_plot_data['k'] = np.linspace(total // 2, 0, 100, endpoint=False, dtype=int)
        hill_plot_data['Gabaix'] = [estimate_gabaix_alpha(data[:k]) for k in hill_plot_data['k']]
        hill_plot_data['Clauset'] = [estimate_clauset_alpha(data[:k]) for k in hill_plot_data['k']]
        fig = px.line(hill_plot_data, x='k', y=['Gabaix', 'Clauset'], labels={'x': 'k', 'value': 'Alpha'})
        st.plotly_chart(fig, use_container_width=True, theme=None)

    fit = pl.Fit(data[:10000])
    xmin = fit.xmin
    data_xmin = data[data >= xmin]
    total_xmin = len(data_xmin)

    alphas = dict()
    cut_off = {'xmin': total_xmin, 'all': total, '100': 100, '1k': 1000, '10k': 10000, '100k': 100000}
    for key, value in cut_off.items():
        if value <= total:
            data_cut = data[:value]
            alphas[f'gabaix_{key}'] = estimate_gabaix_alpha(data_cut)
            alphas[f'clauset_{key}'] = estimate_clauset_alpha(data_cut)
        else:
            alphas[f'gabaix_{key}'] = None
            alphas[f'clauset_{key}'] = None

    st.dataframe({'Total': [total], 'Average': [average], 'Minimum': [minimum], 'Maximum': [maximum], 'Xmin': [xmin], 'Total Xmin': [total_xmin]}, use_container_width=True)
    st.dataframe({'Gabaix Xmin': [alphas['gabaix_xmin']], 'Gabaix All': [alphas['gabaix_all']], 'Gabaix 100': [alphas['gabaix_100']], 'Gabaix 1k': [alphas['gabaix_1k']], 'Gabaix 10k': [alphas['gabaix_10k']], 'Gabaix 100k': [alphas['gabaix_100k']]}, use_container_width=True)
    st.dataframe({'Clauset Xmin': [alphas['clauset_xmin']], 'Clauset All': [alphas['clauset_all']], 'Clauset 100': [alphas['clauset_100']], 'Clauset 1k': [alphas['clauset_1k']], 'Clauset 10k': [alphas['clauset_10k']], 'Clauset 100k': [alphas['clauset_100k']]}, use_container_width=True)
