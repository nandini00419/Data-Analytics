import os 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from pathlib import Path
from data_preprocessing import load_data, load_generation_data

def generate_eda_plot(df: pd.DataFrame, output_dir: Path):
  """ create a path in which all the output is stored which is graph representation"""
  output_dir.mkdir(parents=True, exist_ok=True)
  print(f"existing the output path: {output_dir}")


 # prepare a bar graph of a top 10 electric_vehicle registration according to the states
  #print("Generating the top 10 electric_vehicle registration.")
  plt.figure(figsize=(12,6))
#  Data = Path(__file__).resolve().parent.parent / "Processed_Data"/"clean_data.csv"
#  df = load_generation_data(Data)
  top_10_ev = df.nlargest(10,"ev_count")
  sns.barplot(data = top_10_ev, x="state", y="ev_count")
  plt.title("Top 10 states by ev_registration(2018)", fontsize=16, fontweight='bold')
  plt.xlabel("states", fontsize=12)
  plt.ylabel("ev_count", fontsize=12)
  plt.savefig(output_dir/"sns.barplot.png")
  plt.close()
  print(f"saved: {output_dir/'sns.barplot.png'}")


 #prepare a bar graph of a top 10 total geneartion by the states 
  print("Generating the top 10 electric_generation ")
  plt.figure(figsize=(12,6))
  df = load_generation_data(Data)
  top_10_gen = df.nlargest(10,"total_generation")
  sns.barplot(data = top_10_gen, x="state", y="total_generation")
  plt.title("Top 10 states by total generation_energy(2018)", fontsize=16, fontweight='bold')
  plt.xlabel("states", fontsize=12)
  plt.ylabel("tOtal_generation", fontsize=12)
  plt.savefig(output_dir/"sns.barplot.total_generation.png")
  plt.close()
  print(f"saved: {output_dir/'sns.barplot.total_generation.png'}")


  #prepare a scatter plot between the ev_count & total_generation 
  print("data of a total_generation vs ev_count")
  plt.figure(figsize=(10,7))
  sns.scatterplot(data=df, x="total_generation", y="ev_count", s= 100, color="red", alpha=0.7)
  plt.title("Scatter plot between total generation and ev_count", fontsize=16, fontweight='bold')
  plt.xlabel("total_generation")
  plt.ylabel("ev_count")
  plt.savefig(output_dir/"sns.scatterplot.png")
  plt.close()
  print(f"saved: {output_dir/'sns.scatterplot.png'}")

  # prepare a histplot with a use of a ev registration distribution with a density curve (kde)
  print("distribution of a ev registration accross the states")
  plt.figure(figsize=(10,6))
  sns.histplot(df["ev_count"], kde=True, color="blue", bins=20)
  plt.title("distribution of a ev registration accross the states ",fontsize=16, fontweight='bold')
  plt.xlabel("ev_count", fontsize=12)
  plt.ylabel("frequency",fontsize=12)
  plt.savefig(output_dir/"sns.histplot.png")
  plt.close()
  print(f"saved: {output_dir/'sns.histplot.png'}")


  #prepare a boxplot ev_count to detect a outlier in a taken dataset 
  print("data taken for a boxplot")
  plt.figure(figsize=(8,6))
  sns.boxplot(x=df["ev_count"], color="lightgreen")
  plt.title("box_plot of ev registration", fontsize=16, fontweight='bold')
  plt.xlabel("ev_count", fontsize=12)
  plt.savefig(output_dir/"sns.boxplot.png")
  plt.close()
  print(f"saved: {output_dir/'sns.boxplot.png'}")

#if __name__ == "__main__":
#    Data = Path(__file__).resolve().parent.parent / "Processed_Data"/"clean_data.csv"
#   Output = Path(__file__).resolve().parent.parent / "Output"
#    df = load_generation_data(Data)
#    #print("Columns in dataframe:", df.columns)
#    #print("Data types in dataframe:\n", df.dtypes)
#    #print("Dataframe info:", df.info())
#    #print("Dataframe description:\n", df.describe())
#    #print("Missing values in dataframe:\n", df.isnull().sum())
#    #print("Dataframe head:\n", df.head())
#    #print("data shape:", df.shape)
#    #print("Top 10 states by EV count:\n", df.nlargest(10, "total_generation")[["state", "total_generation"]])
#    #print("data info:", df.info())
#    #print("nlargest state:", df.nlargest(1, "total_generation")[["state", "total_generation"]])
#    print("Data loaded successfully for EDA.")
#    print("Generating EDA plots")
#    generate_eda_plot(df, Output)
#    print("EDA plot generated successfully.")

def generate_renewable_pie(Data: Path, output_dir: Path) -> None:
    print("Loading data for renewable pie chart analysis")
    if not Data.exists():
        raise FileNotFoundError(f"Generation data file not found: {Data}")
    gen = pd.read_excel(Data, skiprows=5, header=None)
    gen.columns = ["year", "state", "type_of_producers", "energy_source", "generation_mwh"]
    gen_2018 = gen[
        (gen["year"] == 2018) & 
        (gen["type_of_producers"] == "Total Electric Power Industry") & 
        (gen["energy_source"] != "Total")
    ].copy()
    print("overview of gen_2018 data:", gen_2018.head())
    
    gen_2018["generation_mwh"] = pd.to_numeric(gen_2018["generation_mwh"].astype(str).str.strip(), errors="coerce").fillna(0)
    renewables = ["Wind", "Wood and Wood Derived Fuels", "Hydroelectric Conventional", "Other Biomass", 
                  "Geothermal", "Solar Thermal and Photovoltaic"]
    gen_2018["is_renewable"] = gen_2018["energy_source"].isin(renewables)
    share = gen_2018.groupby("is_renewable")["generation_mwh"].sum()
    print("Renewable vs Non-Renewable Energy generation calculated successfully.")
    print("overall share of renewable vs non-renewable energy in 2018:", share)
    print("Share of renewable vs non-renewable energy in 2018 calculated successfully.", share)    
    print("Renewable vs Non-Renewable Energy pie chart")

    plt.figure(figsize=(8, 8))
    plt.pie(share, labels=["Non-Renewable", "Renewable"], explode=[0.1,0], colors=["lightblue", "lightgreen"], startangle=90, shadow=True)
    plt.title("Renewable vs Non-Renewable Energy Share (US 2018)", fontsize=16, fontweight='bold')
    plt.savefig(output_dir / "renewable_pie.png")
    plt.close()
    print(f"Saved: {output_dir / 'renewable_pie.png'}")
    print("Renewable pie chart generated successfully.")
#if __name__ == "__main__":
#    generate_renewable_pie(Data = Path(__file__).resolve().parent.parent / "Data"/ "States_Annual_Energy_Generation_Sources_1990_2019.xlsx", output_dir = Path(__file__).resolve().parent.parent / "Output")
#    print("Renewable pie chart generation complete.")
#
#    Data = Path(__file__).resolve().parent.parent / "Data"/ "States_Annual_Energy_Generation_Sources_1990_2019.xlsx"
#    print("Data path for renewable pie chart:", Data)
#    print("data info:", Data.is_file())
#    Output = Path(__file__).resolve().parent.parent / "Output"
#    print("Starting renewable pie chart generation.")
#    generate_renewable_pie(Data, Output)
#    print("Renewable pie chart generation completed successfully.")

def get_summary_stats(df: pd.DataFrame) -> str:
   # Return a summary of EV and generation data.
    stats = df[["ev_count", "total_generation"]].describe().to_string()
    print("Summary statistics for EV count and total generation:", stats)
    print("Summary statistics generated successfully.") 
    print("Summary statistics:\n", stats)
    print("Summary statistics generation completed successfully.")  
    print("Summary statistics generation finished.")
    return stats

def get_correlation(df: pd.DataFrame) -> float:
   # Return the correlation between EV count and total generation.
    corr = df[["ev_count", "total_generation"]].corr().iloc[0, 1]
    print(f"Correlation between EV count and total generation: {corr}")
    print("Correlation calculated successfully.") 
    print("Correlation value:", corr)
    print("Correlation calculation completed successfully.")
    return corr


if __name__ == "__main__":
    Data = Path(__file__).resolve().parent.parent / "Processed_Data"/"clean_data.csv"
    Output = Path(__file__).resolve().parent.parent / "Output"
    df = load_generation_data(Data)
    generate_eda_plot(df, Output)
    summary_stats = get_summary_stats(df)
    correlation = get_correlation(df)
    print("Summary statistics and correlation generated.")
    print("EDA completed successfully.")
    print("Summary Statistics:\n", get_summary_stats(df))
    print("Correlation:", get_correlation(df))
    print("EDA process finished. All outputs saved to:", Output)  
    print("EDA process completed successfully.")
    generate_renewable_pie(Data = Path(__file__).resolve().parent.parent / "Data"/ "States_Annual_Energy_Generation_Sources_1990_2019.xlsx", output_dir = Path(__file__).resolve().parent.parent / "Output")
    print("Renewable pie chart generation complete.")

