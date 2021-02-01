# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table

from pathlib import Path
import pandas as pd
import datetime

from opint_framework.apps.example_app.nlp.pyspark_based.cluster_visualization import plot_time

# Datasets
save_path = Path('/home/luca/PycharmProjects/opint-framework/results/kmeans/2021/02/01')
stats_view = pd.read_json(save_path / "numeric_stats.json", orient='index')
fig = plot_time(save_path / 'count_per_interval.json', save_path=None)

app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/amyoshino/pen/jzXypZ.css'])

app.layout = html.Div(
    html.Div([
        html.Div(
            [
                html.H1(children=f'FTS Failures ({datetime.datetime.now().date()})',
                        className='seven columns'),
                html.Div(
                    children='for support contact luca.clissa2@unibo.it',
                    className='three columns',
                    style={'position': 'relative', 'margin-top': 10}
                ),
            ], className="row"
        ),

        html.Div(
            [
                html.Div([
                    dcc.Graph(
                        id='time-evolution',
                        figure=fig
                    )
                ], className='six columns'
                ),

                html.Div([
                    dash_table.DataTable(
                        id='stats-view',
                        columns=[{"name": i, "id": i} for i in stats_view.columns],
                        data=stats_view.to_dict('records'),
                    )
                ], className='six columns'
                )
            ], className="row"
        )
    ], className='ten columns offset-by-one')
)

if __name__ == '__main__':
    app.run_server(debug=True)