import os
from unittest import result
import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from pathlib import Path
import openpyxl

#clean the columm names by removing extra spaces and converting the names to lowercase
#replace the spces with theunderscores 
def clean_column_names(df):
  df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
#  print("column names cleaned:",df.columns)
  return df 


#convert the columm to the numeric data and handle the non numeric values as well
def to_numeric_clean(series):
    series = series.astype(str).str.replace(',', '').str.strip()
    result = pd.to_numeric(series, errors='coerce')
#    print(f"series converted to numeric, dtype: {result.dtype}")
    return result


# handle the missing vlaues in the data by using the mean, median or mode of the column 
def handle_missing_values(df, strategy='mean'):
  imputer = SimpleImputer(strategy=strategy)
  df_imputed = pd.DataFrame(imputer.fit_transform(df.select_dtypes(include=[np.number])), columns=df.select_dtypes(include=[np.number]).columns)
#  print("missing values handled using strategy:", strategy)
  return df_imputed

# load the data from the csv file and return the dataframe 
def load_generation_data(file_path) -> pd.DataFrame:
  # Check file extension and read accordingly
  if str(file_path).endswith('.xlsx'):
    df = pd.read_excel(file_path, skiprows=5, header=None)
  else:
    df = pd.read_csv(file_path, skiprows=5, header=None, encoding='latin-1')
  df.columns = ["code", "state", "ev_count", "total_generation", "ev_electricity_demand_mwh"]
  df["ev_electricity_demand_mwh"] = to_numeric_clean(df["ev_electricity_demand_mwh"])
#  print("\n gneration data loaded successfully")
#  print("data shape:", df.shape)
#  print("data columns:", df.columns)
#  print("data types:", df.dtypes)
  return df


def load_data(Data_dir: str| Path | None = None) -> pd.DataFrame:
    base = Path(__file__).resolve().parent.parent
    Data = Path(Data_dir) if Data_dir else base / "Data"


    ev_file = Data / "States_Electric_Vehicle_Registrations_2018.xlsx"
    gen_file = Data / "States_Annual_Energy_Generation_Sources_1990_2019.xlsx"
    codes_file = Data / "state_codes.xlsx"
    all_veh_file = Data / "States_All_Vehicle_Registrations_2018.xlsx"
#    print("Data directory:", Data)
#    print("EV file path:", ev_file)
#    print("Generation file path", gen_file)
#    print("State codes file path:", codes_file)
#    print("All vehicle file path:", all_veh_file)

#if __name__ == "__main__":
#   print("Data loading and preprocessing completed successfully.")
   
    
#    print("\n loading data from file:", base)

    if not all(os.path.exists(f) for f in [ev_file, gen_file, codes_file, all_veh_file]):
       raise FileNotFoundError("One or more required data files are missing.")
    
    codes_df = pd.read_excel(codes_file)
    codes_df = clean_column_names(codes_df)
    codes_df["state_name"] = codes_df["state_name"].astype(str).str.strip().str.lower()
    state_map = dict(zip(codes_df["state_name"], codes_df["state_code"]))
#    print("state_map:", state_map)
    
    ev = pd.read_excel(ev_file, skiprows=2)
    ev = clean_column_names(ev)
#    print("ev columns after clean:", ev.columns)
    ev = ev.rename(columns={"state": "state", "registration_count": "ev_count"})
    ev["ev_count"] = to_numeric_clean(ev["ev_count"])
#    print("ev ev_count unique:", ev["ev_count"].unique()[:5])
    ev["state_lower"] = ev["state"].astype(str).str.strip().str.lower()
    ev["state_code"] = ev["state_lower"].map(state_map).str.upper()
    ev = ev.dropna(subset=["state_code"])
#    print("EV data loaded and processed successfully")


    gen = pd.read_excel(gen_file, skiprows=5, header=None)
    gen.columns = ["year", "state", "type_of_producers", "energy_source", "generation_mwh"]
#    print("gen dtypes:", gen.dtypes)
    gen["year"] = pd.to_numeric(gen["year"], errors="coerce")
    gen_2018 = gen[
      (gen["year"] == 2018) &
      (gen["type_of_producers"] == "Total Electric Power Industry") &
      (gen["energy_source"] == "Total")
    ].copy()
    gen_2018["state_code"] = gen_2018["state"].astype(str).str.strip().str.upper()
#    print("Generation data for 2018 loaded and processed successfully")
#    print("gen_2018 shape:", gen_2018.shape)
#    print("ev shape:", ev.shape)


    merged = pd.merge(
    ev[["state_code", "state", "ev_count"]],
    gen_2018[["state_code", "generation_mwh"]],
    on="state_code",
    how="inner"
)

    merged = merged.rename(columns={
    "state_code": "code",
    "generation_mwh": "total_generation"
})
    
    merged = merged.dropna(subset=["ev_count"])
#    print(merged.columns)
#    print("Data merged successfully", merged.shape)
#    print("data preview:", merged.head())
#    print("is ev_count NaN:", merged["ev_count"].isna().sum())
#    print("ev_count values:", merged["ev_count"].head())
#    print("ev_count isna head:", merged["ev_count"].isna().head())
#    print("merged dtypes after merge:", merged.dtypes)
#    print("Merged data shape:", merged.shape)
    merged["ev_count"] = merged["ev_count"].astype(float)
#    print("after astype ev_count:", merged["ev_count"].head())  
    merged["ev_count"] = pd.to_numeric(merged["ev_count"], errors="coerce")
#    print("after to_numeric ev_count:", merged["ev_count"].head())
    merged["ev_electricity_demand_mwh"] = merged["ev_count"] * 3.0
#   print("Electricity demand for EVs calculated successfully")
#    print("final merged head:", merged.head())
#    print("final merged dtypes:", merged.dtypes)
    processed_dir = Path(__file__).resolve().parent.parent / "Processed_Data"
#    print("Saving processed data to:", processed_dir)

    processed_dir.mkdir(exist_ok=True)
    file_path = processed_dir / "clean_data.csv"
    merged.to_csv(file_path, index=False)
#    print("Processed data saved successfully to:", file_path)
    return merged

if __name__ == "__main__":
    print("Data loading and preprocessing completed successfully.")
    processed_data = load_data()
#    print ("ev data preview:", processed_data.head())
#    print("generation data preview:", processed_data.head())
    print("merged data info:", processed_data.info())
    print("merged preview:", processed_data.head()) 
    print("merged overview:\n", processed_data.describe())   
#    print("Processed data info:", processed_data.info())
#    print("Processed data description:", processed_data.describe())
#    print("Missing values in processed data:\n", processed_data.isnull().sum())
#    print("Data types in processed data:\n", processed_data.dtypes)
    print("Data loading and preprocessing completed successfully.")
#    print("Processed data preview:",  processed_data.head())
        


