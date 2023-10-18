from glob import glob

import numpy as np
import pandas as pd
import powerlaw
import plotly.express as px


def compute_gabaix_alpha(data):
    fig = px.scatter(x=np.arange(len(data)) + 0.5, y=data, log_x=True, log_y=True, trendline='ols', trendline_options=dict(log_x=True, log_y=True))
    return -px.get_trendline_results(fig).px_fit_results[0].params[1] + 1


filepaths = glob('data/socialblade/*.csv')
filepaths = sorted([p for p in filepaths if 'history' not in p])

table = pd.DataFrame()
for filepath in filepaths:
    df = pd.read_csv(filepath)
    platform = filepath.split('/')[-1].split('.')[0].capitalize()
    result = {'Platform': platform}
    for column in df.columns:
        if 'rank' in column.lower() or 'id' in column.lower():
            continue
        result['Measure'] = ' '.join([x.capitalize() for x in column.split('_')])
        try:
            data = df[column].astype(float).sort_values(ascending=False).values
            data = data[data > 0]
            drop_ids = np.where(data[1:] / data[:-1] < 0.9)[0]
            drop_ids = drop_ids[drop_ids > len(data) / 2]
            if len(drop_ids) > 0:
                cutoff_id = drop_ids[0]
                data = data[:cutoff_id]
            result['Total Obs'] = len(data)
            result['Max Observed x-value'] = data[0]
            result['Alpha for total Data'] = powerlaw.Fit(data, xmin=data[-1]).alpha
            result['Gabaix for total Data'] = compute_gabaix_alpha(data)
            data = data[:100000]
            result['Alpha for Top 100K Data'] = powerlaw.Fit(data, xmin=data[-1]).alpha
            result['Gabaix for Top 100K Data'] = compute_gabaix_alpha(data)
            xmin = powerlaw.Fit(data[:10000], verbose=False).power_law.xmin
            data = data[data >= xmin]
            result['Alpha for pl.fit data'] = powerlaw.Fit(data, xmin=xmin).alpha
            result['Gabaix for pl.fit data'] = compute_gabaix_alpha(data)
            result['Rank of pl.fit max value'] = len(data)
            table = table._append(result, ignore_index=True)
            table.to_csv('/Users/ykli/Downloads/summary.csv', index=False)
            print(platform, column)
        except Exception as e:
            continue
