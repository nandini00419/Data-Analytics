from pathlib import Path 
import sys
from flask import Flask, render_template, send_from_directory
import logging
import pandas as pd 
from Analysis.predictive import policy_impact_assessment 
from Analysis.policy_impact import main as run_policy_analysis 


logging.basicConfig(
  level = logging.INFO,
   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='template')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/eda')
def eda():
    images = ['sns.barplot.png', 'sns.barplot.total_generation.png', 'sns.scatterplot.png', 'sns.histplot.png', 'sns.boxplot.png']
    return render_template('eda.html', images=images)

@app.route('/time_series')
def time_series():
    images = ['time_series_plot.png', 'trends_analysis.png', 'yoy_changes.png', 'decade_analysis.png', 'arima_forecast.png']
    return render_template('time_series.html', images=images)

@app.route('/correlation')
def correlation():
    images = ['correlation_heatmap.png']
    return render_template('correlation.html', images=images)

@app.route('/geo')
def geo():
    htmls = ['choropleth.html', 'renewable_choropleth.html']
    images = ['renewable_pie.png']
    return render_template('geo.html', htmls=htmls, images=images)

@app.route('/predictive')
def predictive():
    # Assuming predictive has some outputs, but from list, maybe none, or add later
    images = ['clean_energy_forecast.png', 'ev_adoption_predictions.png', 'predictions_plot.png']
    policy_output = policy_impact_assessment()
    return render_template('predictive.html', images=images, policy_output=policy_output)

@app.route('/policy_impact')
def policy_impact():
    results = run_policy_analysis()
    images = ['policy_ev_adoption_trends.png', 'policy_clean_energy_boxplot.png', 'policy_strength_correlation.png']
    return render_template('policy_impact.html',images=images,did_ev=results['did_ev'],
                         did_clean=results['did_clean'],reg_ev_summary=str(results['reg_ev'].summary()),
                         reg_clean_summary=str(results['reg_clean'].summary()),report=results['report'])


@app.route('/Output/<path:filename>')
def serve_output(filename):
    return send_from_directory('Output', filename)

if __name__ == '__main__':
    app.run(debug=True)




