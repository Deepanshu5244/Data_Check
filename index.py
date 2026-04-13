import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import f
from data import *
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
# ==========================================
# 1. HAMPEL FILTER (Univariate Condensing)
# ==========================================
def apply_hampel_filter(df, window_size=5, n_sigmas=3):
    """
    Removes univariate spikes using a rolling median and Median Absolute Deviation (MAD).
    """
    cleaned_df = df.copy()
    
    # Iterate through each sensor column
    for col in cleaned_df.columns:
        # Step 1.1: Calculate the rolling median
        rolling_median = cleaned_df[col].rolling(window=window_size, center=True).median()
        
        # Step 1.2: Calculate the rolling MAD
        rolling_mad = cleaned_df[col].rolling(window=window_size, center=True).apply(
            lambda x: np.median(np.abs(x - np.median(x)))
        )
        
        # Step 1.3: Define the dynamic threshold (1.4826 scales MAD to approx standard deviation)
        threshold = n_sigmas * 1.4826 * rolling_mad
        
        # Step 1.4: Find spikes and replace them with the local rolling median
        outlier_idx = np.abs(cleaned_df[col] - rolling_median) > threshold
        cleaned_df.loc[outlier_idx, col] = rolling_median[outlier_idx]
        
    # Step 1.5: Fill NaN values created at the boundaries of the rolling window
    return cleaned_df.bfill().ffill()

# ==========================================
# 2. PCA FIT & TRANSFORM
# ==========================================
def fit_transform_pca(df, variance_retained=0.95):
    """
    Centers/scales the data and fits the PCA model to calculate Principal Component scores.
    """
    # Step 2.1: Center and scale the data (Required for PCA)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    # Step 2.2: Fit the PCA model retaining the specified variance
    pca_model = PCA(n_components=variance_retained)
    
    # Step 2.3: Calculate the Principal Component scores (projection onto eigenvectors)
    pca_scores = pca_model.fit_transform(scaled_data)
    
    return pca_model, scaled_data, pca_scores

# ==========================================
# 3. PCA RECONSTRUCTION
# ==========================================
def reconstruct_pca(pca_model, pca_scores):
    """
    Converts PCA scores back to the original (scaled) feature space.
    """
    # Step 3.1: Reconstruct the data using the inverse transform mapping
    reconstructed_data = pca_model.inverse_transform(pca_scores)
    
    return reconstructed_data

# ==========================================
# 4. SQUARED PREDICTION ERROR (SPE)
# ==========================================
def calculate_spe(scaled_data, reconstructed_data):
    """
    Calculates reconstruction error and flags anomalies using the Mean + 3*Std threshold.
    """
    # Step 4.1: Calculate SPE for each row (Sum of squared differences)
    spe = np.sum((scaled_data - reconstructed_data) ** 2, axis=1)
    
    # Step 4.2: Calculate the upper threshold boundary
    spe_threshold = np.mean(spe) + (3 * np.std(spe))
    
    # Step 4.3: Flag rows where the SPE exceeds the threshold
    anomalies = spe > spe_threshold
    
    return spe, spe_threshold, anomalies

# ==========================================
# 5. HOTELLING T^2 STATISTIC
# ==========================================
def calculate_hotelling_t2(pca_model, pca_scores, n_samples, alpha=0.95):
    """
    Calculates distance inside PCA space and flags anomalies using the F-distribution limit.
    """
    # Step 5.1: Get the eigenvalues (variance captured by each component)
    eigenvalues = pca_model.explained_variance_
    k = len(eigenvalues) # Number of retained principal components
    
    # Step 5.2: Calculate T^2 statistic for each row (z^2 / lambda)
    # We sum across all retained components for the multivariate score
    t2 = np.sum((pca_scores ** 2) / eigenvalues, axis=1)
    
    # Step 5.3: Calculate the F-distribution threshold limit
    f_stat = f.ppf(alpha, k, n_samples - k)
    t2_threshold = (k * (n_samples - 1) / (n_samples - k)) * f_stat
    
    # Step 5.4: Flag rows where T^2 exceeds the calculated limit
    anomalies = t2 > t2_threshold
    
    return t2, t2_threshold, anomalies

# ==========================================
# MAIN ORCHESTRATION FUNCTION
# ==========================================
def clean_industrial_data(df, method='SPE', variance_retained=0.95, alpha=0.95):
    """
    Main pipeline to clean univariate noise and drop multivariate anomalies.
    Allows choosing between 'SPE' or 'T2' for the anomaly detection method.
    """
    print(f"Starting pipeline. Original dataset shape: {df.shape}")
    
    # Step A: Remove univariate noise
    print("Applying Hampel filter...")
    cleaned_df = apply_hampel_filter(df)
    
    # Step B: Fit PCA and get scores
    print("Fitting PCA model...")
    pca_model, scaled_data, pca_scores = fit_transform_pca(cleaned_df, variance_retained)
    
    # Step C: Execute chosen anomaly detection method
    if method.upper() == 'SPE':
        print("Using SPE (Squared Prediction Error) for anomaly detection...")
        # Reconstruct data (Only needed for SPE)
        reconstructed_data = reconstruct_pca(pca_model, pca_scores)
        # Calculate SPE
        metric_values, threshold, anomalies = calculate_spe(scaled_data, reconstructed_data)
        
    elif method.upper() == 'T2':
        print("Using Hotelling T^2 for anomaly detection...")
        n_samples = df.shape[0]
        # Calculate T^2
        metric_values, threshold, anomalies = calculate_hotelling_t2(pca_model, pca_scores, n_samples, alpha)
        
    else:
        raise ValueError("Invalid method. Choose 'SPE' or 'T2'.")
    
    # Step D: Drop anomalous rows
    anomaly_count = np.sum(anomalies)
    final_df = cleaned_df[~anomalies]
    
    print(f"Dropped {anomaly_count} anomalies using {method.upper()} method.")
    print(f"Cleaned dataset shape: {final_df.shape}")
    
    return final_df,cleaned_df,method

def plot_three_stage(fetched_df, hampel_df, final_df,method):
    """Plot ALL columns → Raw vs Final + SAVE PNG"""

    if fetched_df.empty or final_df is None or final_df.empty:
        print("⚠️ No data to plot")
        return

    # Fix datetime index if needed
    if not isinstance(fetched_df.index, pd.DatetimeIndex):
        if 'time' in fetched_df.columns:
            fetched_df['time'] = pd.to_datetime(fetched_df['time'])
            fetched_df.set_index('time', inplace=True)

            final_df['time'] = pd.to_datetime(final_df['time'])
            final_df.set_index('time', inplace=True)
        else:
            print("⚠️ No datetime index found")

    # Select valid columns
    plot_cols = [col for col in fetched_df.columns if not col.startswith("state__")]

    if not plot_cols:
        print("⚠️ No valid columns to plot")
        return

    n = len(plot_cols)

    fig, axes = plt.subplots(n, 1, figsize=(16, 4 * n), sharex=True)

    if n == 1:
        axes = [axes]

    for i, col in enumerate(plot_cols):
        ax = axes[i]

        # Raw
        ax.plot(fetched_df.index, fetched_df[col],
                label='Raw Data', linewidth=1)

        # Final
        ax.plot(final_df.index, final_df[col],
                label='Final Cleaned', linewidth=2)

        ax.set_title(col)
        ax.set_ylabel("Value")
        ax.grid(True)
        ax.legend()

    # Proper datetime formatting
    import matplotlib.dates as mdates
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.xticks(rotation=45)
    plt.xlabel("Timestamp")

    plt.tight_layout()
   
    filename = f"{TARGET_OUTPUT}_{method}_ALL_TAGS_RAW_vs_FINAL.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    print(f"✅ Graph saved as: {filename}")

    # plt.show()

# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":

    # 🚀 Fetch already processed data from data.py
    print("Fetching data from data.py ...\n")
    fetched_df = get_final_data()          # This is your "raw" data after state filtering

    print(f"Fetched data shape: {fetched_df.shape}")
    print(fetched_df.head())

    # 🧠 Apply PCA cleaning
    final_df, cleaned_df,method = clean_industrial_data(
        fetched_df,
        method='T2',
        variance_retained=0.95,
        alpha=0.95
    )

    # Generate 3-line comparison plot
    print("\nGenerating 3-Stage Comparison Plot...")
    plot_three_stage(fetched_df, cleaned_df, final_df, method)

    print(f"\nFinal Processed Shape: {final_df.shape}")
    print(final_df.head())
    print(final_df.columns)

# if __name__ == "__main__":

#     # 🚀 Fetch already processed data
#     df = get_final_data()

#     print(df.shape)
#     print(df.head())

#     # 🧠 Apply PCA cleaning
#     final_df = clean_industrial_data(
#         df,
#         method='SPE',
#         variance_retained=0.95,
#         alpha=0.95
#     )

#     print(final_df.shape)
