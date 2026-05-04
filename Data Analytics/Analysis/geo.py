import pandas as pd 
import numpy as np 
from pathlib import Path
import plotly.express as px
from data_preprocessing import load_data, load_generation_data

def generate_choropleth(df: pd.DataFrame, output_dir: Path):
  print("generating the choroplet path")

  fig = px.choropleth(
        df,
        locations="code",
        locationmode="USA-states",
        color="ev_count",
        color_continuous_scale=['light pink', 'light blue'],
        hover_name="state",
        scope="usa",
        labels={"ev_count": "EV Registrations"},
        title="2018 US Electric Vehicle Registrations by State"
    )

  print("fig generated successfully")
  print("fig type:", type(fig))

  
  fig.update_layout(
        title_font_size=24,
        margin={"r":0,"t":50,"l":0,"b":0}
    )
    
  output_dir.mkdir(parents=True, exist_ok=True)
  output_path = output_dir / "choropleth.html"
  fig.write_html(str(output_path))
  print(f"Geospatial map saved to {output_path}")
  print("Exiting generate_choropleth")
  print("fig updated layout successfully")
  print("fig updated layout type:", type(fig.update_layout()))

#if __name__ == "__main__":
#    Data_csv = Path(__file__).resolve().parent.parent / "Processed_Data"/"clean_data.csv"
#    Output = Path(__file__).resolve().parent.parent / "Output"
#    df = load_generation_data(Data_csv)
#    print(f"Loading data from {Data_csv}")
#    if Data_csv.exists():
#        df = pd.read_csv(Data_csv)
#        print(f"Data loaded. Shape: {df.shape}")
#        generate_choropleth(df, Output)
#    print("geo graph is formed")
    


def generate_renewable_choropleth(data_path: Path, output_dir: Path):
    """Create a Choropleth map of the US showing renewable energy percentage by state with colors matching the pie chart"""
    print("Generating renewable energy choropleth map")

    
    # Load clean data for state codes
    clean_data_path = Path(__file__).resolve().parent.parent / "Processed_Data" / "clean_data.csv"
    clean_df = pd.read_csv(clean_data_path)

    # Read the Excel of generation data and prepare renewable generation per state (choose 2018)
    df_excel = pd.read_excel(data_path, sheet_name='Net_Generation_1990-2019 Final', header=3)
    df_excel.columns = ['YEAR', 'STATE', 'TYPE OF PRODUCER', 'ENERGY SOURCE', 'GENERATION (Megawatthours)']

    renewable_sources = ['Hydroelectric Conventional', 'Wind', 'Solar Thermal and Photovoltaic',
                         'Wood and Wood Derived Fuels', 'Other Biomass', 'Geothermal', 'Other Gases']

    df_renew = df_excel[df_excel['ENERGY SOURCE'].isin(renewable_sources)]
    df_renew_2018 = df_renew[df_renew['YEAR'] == 2018]
    renewable_by_state = (
        df_renew_2018.groupby('STATE')['GENERATION (Megawatthours)']
        .sum()
        .reset_index()
        .rename(columns={'STATE': 'code', 'GENERATION (Megawatthours)': 'renewable_generation'})
    )

    total_by_state = df_excel[df_excel['YEAR'] == 2018].groupby('STATE')['GENERATION (Megawatthours)'].sum().reset_index()
    total_by_state = total_by_state.rename(columns={'STATE': 'code', 'GENERATION (Megawatthours)': 'total_generation'})
    print("Renewable codes sample:", renewable_by_state['code'].head().tolist())
    print("Total codes sample:", total_by_state['code'].head().tolist())

    renewable_df = pd.merge(renewable_by_state, total_by_state, on='code', how='left')
    renewable_df = pd.merge(renewable_df, clean_df[['state', 'code']], on='code', how='left')    
    print("After merge, shape:", renewable_df.shape)
    print("Missing total_generation:", renewable_df['total_generation'].isna().sum())
    
    # % calculation 
    renewable_df['renewable_percentage'] = (renewable_df['renewable_generation'] / renewable_df['total_generation']) * 100   
    print("Renewable percentage range:", renewable_df['renewable_percentage'].min(), "to", renewable_df['renewable_percentage'].max())
    print("Sample renewable percentages:", renewable_df[['state', 'renewable_percentage']].head())
    
    fig = px.choropleth(
        renewable_df,
        locations='code',
        locationmode='USA-states',
        color='renewable_percentage',
        color_continuous_scale=['lightblue', 'lightgreen'],
        hover_name='state',
        hover_data=['renewable_generation', 'total_generation', 'renewable_percentage'],
        scope='usa',
        labels={'renewable_percentage': 'Renewable Energy %'},
        title='2018 US Renewable Energy Percentage by State'
    )
    
    fig.update_layout(
        title_font_size=24,
        margin={"r":0,"t":50,"l":0,"b":0}
    )
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "renewable_choropleth.html"
    fig.write_html(str(output_path))
    print(f"Renewable geospatial map saved to {output_path}")

if __name__ == "__main__":
    Data_csv = Path(__file__).resolve().parent.parent / "Processed_Data"/"clean_data.csv"
    Data_excel = Path(__file__).resolve().parent.parent / "Data"/"States_Annual_Energy_Generation_Sources_1990_2019.xlsx"
    Output = Path(__file__).resolve().parent.parent / "Output"
    df = load_generation_data(Data_csv)    
    print(f"Loading data from {Data_csv}")
    if Data_csv.exists():
        df = pd.read_csv(Data_csv)
        print(f"Data loaded. Shape: {df.shape}")
        generate_choropleth(df, Output)
        generate_renewable_choropleth(Data_excel, Output)
    else:
        print(f"Error: Data file not found at {Data_csv}")

