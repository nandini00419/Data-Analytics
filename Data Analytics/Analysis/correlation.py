import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import load_generation_data
import os 
import pathlib as path 

def analyze_correlation(data):
  if "ev_count" not in data.columns or "total_generation" not in data.columns:
    print("Error: 'ev_count' or 'total_generation' column not found in the data.")
    return({"error": "required column 'ev_count' or 'total_generation' not found"})
  
  corr_value = float(data["ev_count"].corr(data["total_generation"]))
  print(f"Correlation value: {corr_value}") 
  abs_corr_value = abs(corr_value)
  
  if abs_corr_value > 1.0:
    print("Strong correlation")
    return({"correlation": "strong", "value": corr_value})
  elif abs_corr_value > 0.3:
    print("Moderate correlation")
    return({"correlation": "moderate", "value": corr_value})
  elif abs_corr_value > 0.0:
    print("Weak correlation")
    return({"correlation": "weak", "value": corr_value})
  

def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str):
  output_dir.mkdir(parents=True, exist_ok=True)

  numeric_df = df.select_dtypes(include=['number'])

  if numeric_df.empty:
    print("No numeric columns found for correlation heatmap.")
    return({"error": "no numeric columns found for correlation heatmap"})
  
  corr_matrix = numeric_df.corr()
  plt.figure(figsize=(12,10))
  print("Correlation matrix calculated, generating heatmap",corr_matrix.shape)
  print("corrlation matrix info", corr_matrix.info())
  print("correlation matrix head", corr_matrix.head())
  print("correlation matrix columns", corr_matrix.columns)
  print("correlation matrix index", corr_matrix.index)
  print("correlation matrix values", corr_matrix.values)
  print("correlation matrix is null", corr_matrix.isnull().sum())
  print("correlation matrix describe", corr_matrix.describe())
  print("Correlation matrix calculated, generating heatmap")
  print(corr_matrix)

  sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
  plt.title('Correlation Heatmap')
  plt.savefig(output_dir / 'correlation_heatmap.png')
  plt.close()
  print(f"Correlation heatmap saved to {output_dir / 'correlation_heatmap.png'}")
  return({"message": f"Correlation heatmap saved to {output_dir / 'correlation_heatmap.png'}"}) 
if __name__ == "__main__":
  Data = path.Path(__file__).resolve().parent.parent / "Processed_Data"/"clean_data.csv"
  print(f"Loading data from: {Data}")
  print(f"Data file exists: {Data.exists()}")
  print("data loaded successfully, starting correlation analysis and heatmap generation.")
  Output = path.Path(__file__).resolve().parent.parent / "Output" 
  df = load_generation_data(Data)
  analyze_correlation(df)
  plot_correlation_heatmap(df, Output)
  print("Correlation analysis and heatmap generation completed.")
  

