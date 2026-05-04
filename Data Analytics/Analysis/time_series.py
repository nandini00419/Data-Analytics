import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import numpy as np
import os
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pmdarima as pm


def analyze_time_series(df: pd.DataFrame, output_dir: Path):

  print("time seires analysis: ev & energy production trends")


  # Aggregate functions are used here 
  yearly_generation = df.groupby('year')['generation_mwh'].sum().sort_index()
#  print(f"\nTotal generation by year 5 years:")
#  print(yearly_generation.head())
#  print(f"yearly_generation shape: {yearly_generation.shape}")
#  print(f"yearly_generation index: {yearly_generation.index}")
#  print(f"yearly_generation values: {yearly_generation.values}")
#  print(f"yearly_generation dtypes: {yearly_generation.dtypes}")
#  print(f"yearly_generation head: {yearly_generation.head()}")
#  print(f"yearly_generation tail: {yearly_generation.tail()}")
#  print(f"yearly_generation describe: {yearly_generation.describe()}")
#  print(f"yearly_generation info: {yearly_generation.info()}")    


  # Create DataFrame for analysis
  ts_data = pd.DataFrame({
    'Year': yearly_generation.index,
    'Total Generation (MWh)': yearly_generation.values
  })
  ts_data.set_index('Year', inplace=True)
  
#  print(f"\nTime Series Shape: {ts_data.shape}")
#  print(f"Year Range: {ts_data.index.min()} - {ts_data.index.max()}")
#  print(f"ts_data shape: {ts_data.shape}")
#  print(f"ts_data index: {ts_data.index}")
#  print(f"ts_data values: {ts_data.values}")
#  print(f"ts_data dtypes: {ts_data.dtypes}")
#  print(f"ts_data head: {ts_data.head()}")
#  print(f"ts_data tail: {ts_data.tail()}")
#  print(f"ts_data describe: {ts_data.describe()}")
#  print(f"ts_data info: {ts_data.info()}")
#  print(f"ts_data columns: {ts_data.columns}")
#  print(f"ts_data index: {ts_data.index}")
#  print(f"ts_data values: {ts_data.values}")
#  print(f"ts_data dtypes: {ts_data.dtypes}")
#  print(f"ts_data head: {ts_data.head()}")
#  print(f"ts_data tail: {ts_data.tail()}")
#  print(f"ts_data describe: {ts_data.describe()}")
#  print(f"ts_data info: {ts_data.info()}")
  
  # Calculate yoy
  ts_data['YoY Change (MWh)'] = ts_data['Total Generation (MWh)'].diff()
  ts_data['YoY % Change'] = ts_data['Total Generation (MWh)'].pct_change() * 100
  
#  print("\nYear-over-Year Changes (first 10 years):")
#  print(ts_data[['Total Generation (MWh)', 'YoY Change (MWh)', 'YoY % Change']].head(10))
#  print(f"ts_data YoY Change (MWh): {ts_data['YoY Change (MWh)']}")
#  print(f"ts_data YoY % Change: {ts_data['YoY % Change']}")
#  print(f"ts_data YoY Change (MWh) shape: {ts_data['YoY Change (MWh)'].shape}")
#  print(f"ts_data YoY % Change shape: {ts_data['YoY % Change'].shape}")
#  print(f"ts_data YoY Change (MWh) head: {ts_data['YoY Change (MWh)'].head()}")
#  print(f"ts_data YoY % Change head: {ts_data['YoY % Change'].head()}")
#  print(f"ts_data YoY Change (MWh) tail: {ts_data['YoY Change (MWh)'].tail()}")
#  print(f"ts_data YoY % Change tail: {ts_data['YoY % Change'].tail()}")
#  print(f"ts_data YoY Change (MWh) describe: {ts_data['YoY Change (MWh)'].describe()}")
#  print(f"ts_data YoY % Change describe: {ts_data['YoY % Change'].describe()}")
#  print(f"ts_data YoY Change (MWh) info: {ts_data['YoY Change (MWh)'].info()}")
#  print(f"ts_data YoY % Change info: {ts_data['YoY % Change'].info()}")
#  print(f"ts_data YoY Change (MWh) index: {ts_data['YoY Change (MWh)'].index}")
#  print(f"ts_data YoY % Change index: {ts_data['YoY % Change'].index}")
#  print(f"ts_data YoY Change (MWh) values: {ts_data['YoY Change (MWh)'].values}")
#  print(f"ts_data YoY % Change values: {ts_data['YoY % Change'].values}")
#  print(f"ts_data YoY Change (MWh) dtypes: {ts_data['YoY Change (MWh)'].dtypes}")
  
  # growth peroid 
  growth_years = ts_data[ts_data['YoY Change (MWh)'] > 0]
  decline_years = ts_data[ts_data['YoY Change (MWh)'] < 0]
  
  print(f"\nGrowth Years: {len(growth_years)}")
  print(f"Decline Years: {len(decline_years)}")
  print(f"growth_years shape: {growth_years.shape}")
  print(f"decline_years shape: {decline_years.shape}")
  print(f"growth_years index: {growth_years.index}")
  print(f"decline_years index: {decline_years.index}")
  print(f"growth_years values: {growth_years.values}")
  print(f"decline_years values: {decline_years.values}")
  print(f"growth_years dtypes: {growth_years.dtypes}")
  
  if len(growth_years) > 0:
    avg_growth = growth_years['YoY % Change'].mean()
    max_growth_year = growth_years['YoY % Change'].idxmax()
    max_growth_pct = growth_years['YoY % Change'].max()
    print(f"Average Growth Rate: {avg_growth:.2f}%")
    print(f"Peak Growth Year: {max_growth_year} ({max_growth_pct:.2f}%)")
    print(f"avg_growth: {avg_growth}")
    print(f"max_growth_year: {max_growth_year}")
    print(f"max_growth_pct: {max_growth_pct}")
  

  plt.figure(figsize=(14, 6))
  plt.plot(ts_data.index, ts_data['Total Generation (MWh)'], linewidth=2, color='darkblue')
  plt.title('Total Energy Generation Over Time (1990-2019)', fontsize=16, fontweight='bold')
  plt.xlabel('Year', fontsize=12)
  plt.ylabel('Total Generation (MWh)', fontsize=12)
  plt.grid(True)
  plt.savefig(output_dir / 'time_series_plot.png')
  print("Saved: time_series_plot.png")
  plt.close()
  
  # graph of yoy 
  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
  colors = ['green' if x > 0 else 'red' for x in ts_data['YoY Change (MWh)'].dropna()]
  ax1.bar(ts_data.index[1:], ts_data['YoY Change (MWh)'].dropna(), color=colors, alpha=0.7)
  ax1.set_title('Year-over-Year Change in Total Generation (Absolute)', fontsize=14, fontweight='bold')
  ax1.set_ylabel('Change (MWh)', fontsize=11)
  ax1.grid(True, alpha=0.3, axis='y')
  ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
  
  # YoY percentage change
  colors = ['green' if x > 0 else 'red' for x in ts_data['YoY % Change'].dropna()]
  ax2.bar(ts_data.index[1:], ts_data['YoY % Change'].dropna(), color=colors, alpha=0.7)
  ax2.set_title('Year-over-Year Change in Total Generation (Percentage)', fontsize=14, fontweight='bold')
  ax2.set_xlabel('Year', fontsize=11)
  ax2.set_ylabel('% Change', fontsize=11)
  ax2.grid(True, alpha=0.3, axis='y')
  ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
  
  plt.tight_layout()
  plt.savefig(output_dir / 'yoy_changes.png', dpi=300, bbox_inches='tight')
  print("Saved: yoy_changes.png")
  plt.close()
  

  # Since this is annual data, we'll use moving averages to identify trends
  ts_data['MA_3Year'] = ts_data['Total Generation (MWh)'].rolling(window=3, center=True).mean()
  ts_data['MA_5Year'] = ts_data['Total Generation (MWh)'].rolling(window=5, center=True).mean()
#  print(f"ts_data MA_3Year: {ts_data['MA_3Year']}")
#  print(f"ts_data MA_5Year: {ts_data['MA_5Year']}")
#  print(f"ts_data MA_3Year shape: {ts_data['MA_3Year'].shape}")
#  print(f"ts_data MA_5Year shape: {ts_data['MA_5Year'].shape}")
#  print(f"ts_data MA_3Year head: {ts_data['MA_3Year'].head()}")
#  print(f"ts_data MA_5Year head: {ts_data['MA_5Year'].head()}")
#  print(f"ts_data MA_3Year tail: {ts_data['MA_3Year'].tail()}")
#  print(f"ts_data MA_5Year tail: {ts_data['MA_5Year'].tail()}")
  
  # Calculate linear trend
  x = np.arange(len(ts_data))
  y = ts_data['Total Generation (MWh)'].values
  z = np.polyfit(x, y, 1)
  p = np.poly1d(z)
  ts_data['Linear Trend'] = p(x)
#  print(f"ts_data Linear Trend: {ts_data['Linear Trend']}")
#  print(f"ts_data Linear Trend shape: {ts_data['Linear Trend'].shape}")
#  print(f"ts_data Linear Trend head: {ts_data['Linear Trend'].head()}")
#  print(f"ts_data Linear Trend tail: {ts_data['Linear Trend'].tail()}")
#  print(f"ts_data Linear Trend describe: {ts_data['Linear Trend'].describe()}")
#  print(f"ts_data Linear Trend info: {ts_data['Linear Trend'].info()}")
#  print(f"ts_data Linear Trend index: {ts_data['Linear Trend'].index}")
#  print(f"ts_data Linear Trend values: {ts_data['Linear Trend'].values}")
#  print(f"ts_data Linear Trend dtypes: {ts_data['Linear Trend'].dtypes}")
  
  trend_slope = z[0]
  trend_direction = "increasing" if trend_slope > 0 else "decreasing"
  print(f"\nLong-term Trend: {trend_direction}")
  print(f"Trend Slope: {trend_slope:,.0f} MWh/year")
  print(f"Total Change (1990-2019): {ts_data['Total Generation (MWh)'].iloc[-1] - ts_data['Total Generation (MWh)'].iloc[0]:,.0f} MWh")
  
  # Moving Averages
  plt.figure(figsize=(14, 7))
  plt.plot(ts_data.index, ts_data['Total Generation (MWh)'], label='Actual Data', alpha=0.7, linewidth=1.5)
  plt.plot(ts_data.index, ts_data['MA_3Year'], label='3-Year MA', linewidth=2.5, color='orange')
  plt.plot(ts_data.index, ts_data['MA_5Year'], label='5-Year MA', linewidth=2.5, color='red')
  plt.plot(ts_data.index, ts_data['Linear Trend'], label='Linear Trend', linewidth=2, linestyle='--', color='green')
  
  plt.title('Long-Term Trends in Energy Generation', fontsize=16, fontweight='bold')
  plt.xlabel('Year', fontsize=12)
  plt.ylabel('Total Generation (MWh)', fontsize=12)
  plt.legend(fontsize=11, loc='best')
  plt.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.savefig(output_dir / 'trends_analysis.png', dpi=300, bbox_inches='tight')
  print("Saved: trends_analysis.png")
  plt.close()
  
  # decline 
  ts_data['Period'] = pd.cut(ts_data.index, bins=3, labels=['1990s', '2000s', '2010s'])
  period_summary = ts_data.groupby('Period', observed=True)['Total Generation (MWh)'].agg(['mean', 'min', 'max', 'std'])
  
  print("\nGeneration by Decade:")
  print(period_summary)
  
  # Decade Comparison
  fig, axes = plt.subplots(1, 2, figsize=(14, 5))
#  print(f"fig: {fig}")
#  print(f"axes: {axes}")
#  print(f"axes[0]: {axes[0]}")
#  print(f"axes[1]: {axes[1]}")
#  print(f"axes[0] title: {axes[0].title}")
#  print(f"axes[1] title: {axes[1].title}")
 
  
  # Box plot by decade
  ts_data.boxplot(column='Total Generation (MWh)', by='Period', ax=axes[0])
  axes[0].set_title('Distribution of Generation by Decade')
  axes[0].set_ylabel('Total Generation (MWh)')
  axes[0].set_xlabel('Decade')
  
  # Bar plot of decade averages
  period_means = ts_data.groupby('Period', observed=True)['Total Generation (MWh)'].mean()
  axes[1].bar(range(len(period_means)), period_means.values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
  axes[1].set_xticks(range(len(period_means)))
  axes[1].set_xticklabels(period_means.index)
  axes[1].set_title('Average Generation by Decade')
  axes[1].set_ylabel('Average Generation (MWh)')
  axes[1].grid(True, alpha=0.3, axis='y')
  
  plt.tight_layout()
  plt.savefig(output_dir / 'decade_analysis.png', dpi=300, bbox_inches='tight')
  print("Saved: decade_analysis.png")
  plt.close()
  
  # ARIMA Forecasting
 
  try:
    model = pm.auto_arima(ts_data['Total Generation (MWh)'], stepwise=True)
    print(f"\nARIMA Model: ARIMA{model.order}")
    print(f"AIC: {model.aic():.2f}")
    print(f"BIC: {model.bic():.2f}")
    
    # prediction next 10 years
    forecast = model.predict(n_periods=10)
    forecast_years = np.arange(ts_data.index.max() + 1, ts_data.index.max() + 11)
    
    #ARIMA- plot 
    plt.figure(figsize=(14, 7))
    plt.plot(ts_data.index, ts_data['Total Generation (MWh)'], label='Historical Data', marker='o', linewidth=2, markersize=6)
    plt.plot(forecast_years, forecast, label='Forecast', marker='s', linewidth=2, markersize=6, color='red')
    
    plt.title('Energy Generation: Historical Data & Forecast (ARIMA)', fontsize=16, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Total Generation (MWh)', fontsize=12)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'arima_forecast.png', dpi=300, bbox_inches='tight')
    print("Saved: arima_forecast.png")
    plt.close()
    
    print(f"\n10-Year Forecast (2020-2029):")
    for year, value in zip(forecast_years, forecast):
      print(f"  {year}: {value:,.0f} MWh")
  
  except Exception as e:
    print(f"ARIMA model error: {str(e)}")
    
  

if __name__ == "__main__":
  Data = Path(__file__).resolve().parent.parent / 'Data' / 'States_Annual_Energy_Generation_Sources_1990_2019.xlsx'
  Output = Path(__file__).resolve().parent.parent / 'Output'
  df = pd.read_excel(Data, skiprows=4)
  df.columns = ['year', 'state', 'sector', 'source', 'generation_mwh']
  df['year'] = pd.to_numeric(df['year'], errors='coerce')
  df['generation_mwh'] = pd.to_numeric(df['generation_mwh'], errors='coerce')
  df = df.dropna(subset=['year', 'generation_mwh'])
  df['year'] = df['year'].astype(int)
  analyze_time_series(df, Output)
  print("time series analysis completed successfully.")
#  print("time series analysis completed successfully.", df)
#  print("time series analysis completed successfully.", df.head())
#  print("time series analysis completed successfully.", df.tail())
#  print("time series analysis completed successfully.", df.describe())
#  print("time series analysis completed successfully.", df.info())
#  print("time series analysis completed successfully.", df.columns)
#  print("time series analysis completed successfully.", df.index)
#  print("time series analysis completed successfully.", df.values)
#  print("time series analysis completed successfully.", df.dtypes)
#  print("time series analysis completed successfully.", df.shape)
#  print("time series analysis completed successfully.", df.head())
  










  