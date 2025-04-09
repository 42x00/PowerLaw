import numpy as np
import pandas as pd
import powerlaw as pl
import streamlit as st
import plotly.express as px
import statsmodels.api as sm

def estimate_power_law_alpha(data):
    """
    Estimate the power-law exponent alpha using the Gabaix rank-1/2 method.
    
    Parameters:
    - data: list or numpy array of positive numerical values
    
    Returns:
    - alpha_estimate: estimated power-law exponent alpha
    - se: standard error of the alpha estimate
    """
    # Convert data to numpy array and sort in descending order
    sorted_data = np.sort(np.array(data))[::-1]
    n = len(sorted_data)
    
    # Assign ranks: 1 for the largest value, 2 for the next, etc.
    ranks = np.arange(1, n + 1)
    
    # Adjust ranks by subtracting 0.5 as per Gabaix method
    adjusted_ranks = ranks - 0.5
    
    # Compute logarithms of adjusted ranks and sizes
    log_adjusted_ranks = np.log(adjusted_ranks)
    log_sizes = np.log(sorted_data)
    
    # Prepare independent variable with a constant term for OLS
    X = sm.add_constant(log_sizes)
    
    # Perform OLS regression: log(rank - 0.5) = a + b * log(size)
    model = sm.OLS(log_adjusted_ranks, X).fit()
    
    # The slope (params[1]) is -b, where b is the estimate of alpha
    alpha_estimate = -model.params[1]
    
    # Compute standard error as recommended by Gabaix: sqrt(2/n) * alpha_estimate
    se = (2 / n) ** 0.5 * alpha_estimate
    
    return alpha_estimate, se

st.title('Hill Plot App')

uploaded_file = st.file_uploader("File")
if uploaded_file is None:
    st.stop()
df = pd.read_csv(uploaded_file)
st.dataframe(df, use_container_width=True)

column = st.selectbox('Column', [''] + list(df.columns))
if column == '':
    st.stop()

data = df[column].values
data = data[data > 0]

# Step 2: Sort data in descending order
data_sorted = np.sort(data)[::-1]  # X_{(1)} >= X_{(2)} >= ... >= X_{(n)}
n_samples = len(data_sorted)

# Step 3: Compute Hill estimator for various k values
k_values = np.arange(10, n_samples // 2 + 1, 10)  # Range of k from 10 to n/2, step 10
alpha_hill = []
alpha_clauset = []
alpha_gabaix = []

for k in k_values:
    top_k = data_sorted[:k]       # Top k order statistics
    X_k1 = data_sorted[k]         # X_{(k+1)}, the (k+1)th order statistic
    log_terms = np.log(top_k / X_k1)  # Log ratios
    alpha_hat = k / np.sum(log_terms)  # Hill estimator formula
    alpha_hill.append(alpha_hat)

    fit = pl.Fit(top_k, xmin=X_k1)
    alpha_clauset.append(fit.alpha - 1)

    alpha_gabaix.append(estimate_power_law_alpha(top_k)[0])

result = pd.DataFrame({
    'k': k_values,
    'Hill Estimator': alpha_hill,
    'Clauset Estimator': alpha_clauset,
    'Gabaix Estimator': alpha_gabaix
})

# Step 4: Plotting the results
fig = px.line(result, x='k', y=['Hill Estimator', 'Clauset Estimator', 'Gabaix Estimator'], labels={'value': 'α'})

st.plotly_chart(fig)