"""
Customer Segmentation and Behavioral Analysis Tool
Author: Data Analytics Team
Description: Segments customer demographic and spending profiles using K-Means Clustering from a CSV file.
"""

import os
import warnings

# ---------------------------------------------------------
# STEP 1: ENVIRONMENT CONFIGURATION & SYSTEM SILENCING
# ---------------------------------------------------------
# Force joblib to use logical cores and suppress deprecated Windows wmic backend utility warnings
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# STEP 2: EXTERNAL DATA INGESTION
# ---------------------------------------------------------
def load_external_dataset(file_path):
    """Loads the customer transaction dataset from a local CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"[ERROR] Could not find '{file_path}' in the current working directory.\n"
            f"Please ensure you place the downloaded CSV file in the same folder as this script."
        )
    
    print(f"[INFO] Successfully located and importing '{file_path}'...")
    dataset = pd.read_csv(file_path)
    return dataset

# ---------------------------------------------------------
# STEP 3: MACHINE LEARNING PIPELINE (K-MEANS)
# ---------------------------------------------------------
def execute_customer_segmentation(df):
    """Processes features and applies an optimized K-Means Clustering pipeline."""
    print("[INFO] Extracting feature vectors and normalizing scale matrices...")
    
    # Selecting numerical features for behavioral clustering
    features = df[['Annual_Income_k$', 'Spending_Score_1_100']]
    
    # Standardizing feature scales for unbiased Euclidean distance calculations
    feature_scaler = StandardScaler()
    scaled_features = feature_scaler.fit_transform(features)
    
    print("[INFO] Fitting K-Means Clustering model (Hyperparameter k=3)...")
    clustering_model = KMeans(n_clusters=3, init='k-means++', random_state=42)
    df['Cluster_ID'] = clustering_model.fit_predict(scaled_features)
    
    # Mapping statistical cluster IDs to target marketing segments
    segment_dictionary = {
        0: 'Frugal / Budget Shoppers',
        1: 'Premium / High-Value Targets',
        2: 'Average / Mainstream Consumers'
    }
    df['Customer_Segment'] = df['Cluster_ID'].map(segment_dictionary)
    return df

# ---------------------------------------------------------
# STEP 4: DATA VISUALIZATION EXPORT
# ---------------------------------------------------------
def generate_segment_visuals(df):
    """Generates high-fidelity scatter plots tracking demographic cluster splits."""
    print("[INFO] Generating high-fidelity visualization matrix...")
    plt.figure(figsize=(11, 6.5))
    sns.set_theme(style="whitegrid")
    
    # Rendering spatial cluster scatter layout
    sns.scatterplot(
        data=df, 
        x='Annual_Income_k$', 
        y='Spending_Score_1_100', 
        hue='Customer_Segment', 
        palette='Set1', 
        s=120, 
        alpha=0.85,
        edgecolor='w'
    )
    
    # Enhancing metadata labels and presentation format
    plt.title('Customer Demographics & Behavior Segmentation Matrix', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Annual Income (k$)', fontsize=11)
    plt.ylabel('Spending Score (Scale 1-100)', fontsize=11)
    plt.legend(title='Identified Segments', loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True)
    
    plt.tight_layout()
    output_filename = 'customer_segments_chart.png'
    plt.savefig(output_filename, dpi=300)
    print(f"[SUCCESS] Analytical chart successfully exported as '{output_filename}'")
    plt.show()

# ---------------------------------------------------------
# STEP 5: CORE MAIN EXECUTIVE RUNTIME LOOP
# ---------------------------------------------------------
if __name__ == "__main__":
    print("=== STARTING CUSTOMER SEGMENTATION CORE ENGINE ===")
    
    # Define local data file name target
    TARGET_DATA_FILE = "customer_demographics_data.csv"
    
    try:
        # Step A: Load CSV records
        raw_data = load_external_dataset(TARGET_DATA_FILE)
        
        # Step B: Pass dataset through machine learning model algorithm
        segmented_data = execute_customer_segmentation(raw_data)
        
        # Step C: Print data profiling insights table to console
        print("\n=== ANALYTICAL SEGMENT DATA PROFILE SUMMARY ===")
        summary_metrics = segmented_data.groupby('Customer_Segment')[['Age', 'Annual_Income_k$', 'Spending_Score_1_100']].mean()
        print(summary_metrics.round(2).to_string())
        print("===============================================\n")
        
        # Step D: Save output charts locally
        generate_segment_visuals(segmented_data)
        print("=== PIPELINE RUN COMPLETE AND READY FOR SUBMISSION ===")
        
    except Exception as error_msg:
        print(error_msg)