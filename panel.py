import numpy as np
import pandas as pd
import powerlaw
import plotly.express as px


def get_slope(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return px.get_trendline_results(px.scatter(df, trendline='ols')).px_fit_results.iloc[0].params[1]


def compare_distributions(x):
    try:
        return x.distribution_compare('power_law', 'truncated_power_law')[0]
    except:
        return np.nan


def get_results(df, suffix=''):
    ret = dict()
    # gab
    gab_df = px.get_trendline_results(
        px.scatter(df, x=column, y='rank', color=groupby, log_x=True, log_y=True, trendline='ols',
                   trendline_options=dict(log_x=True, log_y=True))).set_index(groupby)['px_fit_results'].apply(
        lambda x: -x.params[1] + 1)
    ret[f'gab{suffix}_trend'] = get_slope(gab_df)
    ret[f'gab{suffix}_mean'] = gab_df.mean()

    # pl & r
    fits = df.groupby(groupby)[column].apply(lambda x: powerlaw.Fit(x, xmin=x.min()))
    pl_df = fits.apply(lambda x: x.power_law.alpha)
    ret[f'pl{suffix}_trend'] = get_slope(pl_df)
    ret[f'pl{suffix}_mean'] = pl_df.mean()
    r_df = fits.apply(lambda x: compare_distributions(x))
    ret[f'r{suffix}_trend'] = get_slope(r_df)
    ret[f'r{suffix}_mean'] = r_df.mean()
    return ret


path = '/Users/ykli/Downloads/panel.csv'
panel = pd.read_csv(path)

for i, row in panel.iterrows():
    if row['status'] == 'done':
        continue
    filepath, groupby, column = row['filepath'], row['groupby'], row['column']
    print(filepath, groupby, column)
    result = dict()

    # df
    df = pd.read_csv(filepath)[[groupby, column]]
    df[groupby] = df[groupby].astype(str)
    df[column] = df[column].astype(float)
    df = df[df[column] > 0]
    df = df.sort_values(by=[groupby, column], ascending=[True, False])
    df['rank'] = df.groupby(groupby).cumcount() + 0.5

    # year
    result['period'] = len(df[groupby].unique())

    # max
    max_df = df.groupby(groupby).max()
    result['max_trend'] = get_slope(max_df)
    result['max_mean'] = max_df[column].mean()

    # gab, pl, r
    result.update(get_results(df))

    # xmin
    df = df.groupby(groupby)[column].nlargest(10000).droplevel(1).reset_index()
    df['rank'] = df.groupby(groupby).cumcount() + 0.5
    mask = df.groupby(groupby)[column].transform(lambda x: x >= powerlaw.Fit(x, verbose=False).power_law.xmin)
    df = df[mask]
    xmin_df = df.groupby(groupby).min()
    result['xmin_trend'] = get_slope(xmin_df)
    result['xmin_mean'] = xmin_df[column].mean()

    # obs
    obs_df = df.groupby(groupby).count()
    result['obs_trend'] = get_slope(obs_df)
    result['obs_mean'] = obs_df[column].mean()

    # gab_fit, pl_fit, r_fit
    result.update(get_results(df, suffix='_fit'))

    # update
    for k, v in result.items():
        panel.at[i, k] = v
    panel.at[i, 'status'] = 'done'
    panel.to_csv(path, index=False)
