import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import f
# from data import *
from datafatch import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from numpy.lib.stride_tricks import sliding_window_view

# ==========================================
# 1. HAMPEL FILTER (Univariate Condensing)
# ==========================================

def apply_hampel_filter(df, window_size=5, n_sigmas=3):
    # Padding size for center=True alignment
    pad_size = window_size // 2
    
    # Convert DataFrame to a 2D NumPy array
    arr = df.to_numpy()
    
    # Pad the top and bottom with NaNs to handle the edges (simulates center=True)
    padded_arr = np.pad(arr, pad_width=((pad_size, pad_size), (0, 0)), 
                        mode='constant', constant_values=np.nan)
    
    # Create an efficient 3D view of the sliding windows
    # Shape becomes: (n_rows, n_columns, window_size)
    windows = sliding_window_view(padded_arr, window_shape=window_size, axis=0)
    
    # 1. Vectorized Rolling Median across the 3rd axis (window_size)
    rolling_median = np.nanmedian(windows, axis=-1)
    
    # 2. Vectorized Rolling MAD
    # Expand rolling_median dimensions to allow broadcasting against the 3D windows array
    abs_deviations = np.abs(windows - rolling_median[..., np.newaxis])
    rolling_mad = np.nanmedian(abs_deviations, axis=-1)
    
    # Calculate dynamic threshold
    threshold = n_sigmas * 1.4826 * rolling_mad
    
    # Find indices where the difference exceeds the threshold
    # Note: Using np.where avoids raising warnings on NaN comparisons
    outliers = np.abs(arr - rolling_median) > threshold
    
    # Replace outliers with the localized rolling median
    cleaned_arr = np.where(outliers, rolling_median, arr)
    
    # Reconstruct the pandas DataFrame and fill the edge NaNs
    cleaned_df = pd.DataFrame(cleaned_arr, index=df.index, columns=df.columns)
    
    return cleaned_df.bfill().ffill()


# ==========================================
# 2. PCA FIT & TRANSFORM
# ==========================================
def fit_transform_pca(df, variance_retained=0.95):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    pca_model = PCA(n_components=variance_retained)
    pca_scores = pca_model.fit_transform(scaled_data)

    return pca_model, scaled_data, pca_scores


# ==========================================
# 3. PCA RECONSTRUCTION
# ==========================================
def reconstruct_pca(pca_model, pca_scores):
    return pca_model.inverse_transform(pca_scores)


# ==========================================
# 4. SPE
# ==========================================
def calculate_spe(scaled_data, reconstructed_data):
    spe = np.sum((scaled_data - reconstructed_data) ** 2, axis=1)
    spe_threshold = np.mean(spe) + (3 * np.std(spe))
    anomalies = spe > spe_threshold
    return spe, spe_threshold, anomalies


# ==========================================
# 5. HOTELLING T²
# ==========================================
# def calculate_hotelling_t2(pca_model, pca_scores, n_samples, alpha=0.95):
#     eigenvalues = pca_model.explained_variance_
#     k = len(eigenvalues)

#     t2 = np.sum((pca_scores ** 2) / eigenvalues, axis=1)

#     f_stat = f.ppf(alpha, k, n_samples - k)
#     t2_threshold = (k * (n_samples - 1) / (n_samples - k)) * f_stat

#     anomalies = t2 > t2_threshold

#     return t2, t2_threshold, anomalies

def calculate_hotelling_t2(pca_model, pca_scores, n_samples, alpha=0.99):
    eigenvalues = pca_model.explained_variance_
    k = len(eigenvalues)

    # Calculate T² statistic
    t2 = np.sum((pca_scores ** 2) / eigenvalues, axis=1)

    f_stat = f.ppf(alpha, k, n_samples - k)
    
    # Main multiplier (standard formula)
    multiplier = (k * (n_samples - 1)) / (n_samples - k)
    
    # Add a small relaxation factor (1.1 to 1.5) to make threshold higher
    relaxation_factor = 1   # ← Tune this (1.0 = original, 1.3 = much looser)
    
    t2_threshold = multiplier * f_stat * relaxation_factor

    # Flag anomalies
    anomalies = t2 > t2_threshold

    return t2, t2_threshold, anomalies


# ==========================================
# MAIN ORCHESTRATION (UPDATED)
# ==========================================
def clean_industrial_data(df, variance_retained=0.95, alpha=0.95):
    print(f"Starting pipeline. Original dataset shape: {df.shape}")

    # Step A: Hampel
    print("Applying Hampel filter...")
    cleaned_df = apply_hampel_filter(df)
    print(f"Shape after Hampel filter: {cleaned_df.shape}")
    # Step B: PCA
    print("Fitting PCA model...")
    pca_model, scaled_data, pca_scores = fit_transform_pca(cleaned_df, variance_retained)

    # ================================
    # SPE
    # ================================
    print("Using SPE (Squared Prediction Error)...")
    reconstructed_data = reconstruct_pca(pca_model, pca_scores)
    spe_values, spe_threshold, spe_anomalies = calculate_spe(scaled_data, reconstructed_data)

    # ================================
    # T²
    # ================================
    print("Using Hotelling T²...")
    n_samples = df.shape[0]
    t2_values, t2_threshold, t2_anomalies = calculate_hotelling_t2(
        pca_model, pca_scores, n_samples, alpha
    )

    # Step D: Drop anomalies separately
    final_spe_df = cleaned_df[~spe_anomalies]
    final_t2_df = cleaned_df[~t2_anomalies]

    print(f"Dropped {np.sum(spe_anomalies)} anomalies using SPE")
    print(f"Dropped {np.sum(t2_anomalies)} anomalies using T²")

    return final_spe_df, final_t2_df, cleaned_df


# ==========================================
# PLOT FUNCTION (UPDATED)
# ==========================================
def plot_three_stage(fetched_df, spe_df, t2_df, tag_desc_map):

    plot_cols = [col for col in fetched_df.columns if not col.startswith("state__")]

    n = len(plot_cols)
    fig, axes = plt.subplots(n, 1, figsize=(16, 4 * n), sharex=True)

    if n == 1:
        axes = [axes]

    for i, col in enumerate(plot_cols):
        ax = axes[i]

        ax.plot(fetched_df.index, fetched_df[col], label='Raw Data', linewidth=1)
        ax.plot(spe_df.index, spe_df[col], label='SPE Cleaned', linewidth=2)
        ax.plot(t2_df.index, t2_df[col], label='T² Cleaned', linewidth=2)

        desc = tag_desc_map.get(col, "No Description")

        ax.set_title(f"{col} | {desc} |raw: {len(fetched_df)} |spe: {len(spe_df)} |t2: {len(t2_df)}", fontsize=12)
        ax.set_ylabel("Value")
        ax.grid(True)
        ax.legend()

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.xticks(rotation=45)
    plt.xlabel("Timestamp")

    plt.tight_layout()
    filename = f"loadtags_RAW_vs_SPE_vs_T2.png"
    # filename = f"{TARGET_OUTPUT}_RAW_vs_SPE_vs_T2.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    print(f"✅ Graph saved as: {filename}")


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":

    print("Fetching data from data.py ...\n")
    fetched_df, tag_info = get_final_data()  

    print(f"Fetched data shape: {fetched_df.shape}")
    print(fetched_df.head())

    final_spe_df, final_t2_df, cleaned_df = clean_industrial_data(
        fetched_df,
        variance_retained=0.95,
        alpha=0.95
    )

    print("\nGenerating Comparison Plot...")

    tag_desc_map = {
        item["tag"]: item.get("description", "")
        for item in tag_info
    }

    plot_three_stage(fetched_df, final_spe_df, final_t2_df, tag_desc_map)

    print("\nFinal Shapes:")
    print("SPE:", final_spe_df.shape)
    print("T2 :", final_t2_df.shape)