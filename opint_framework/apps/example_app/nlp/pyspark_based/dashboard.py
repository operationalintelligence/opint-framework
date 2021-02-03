# -*- coding: utf-8 -*-
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objects as go

from opint_framework.apps.example_app.nlp.pyspark_based.cluster_visualization import plot_time

# TODO: link cluster colors in time evolution plot and pie charts
# Datasets
save_path = Path('/home/luca/PycharmProjects/opint-framework/results/kmeans/2021/02/01')
stats_view = pd.read_json(save_path / "numeric_stats.json", orient='index')
fig = plot_time(save_path / 'count_per_interval.json', save_path=None, show=False)
size_pie = go.Figure(
    go.Pie(labels=stats_view.prediction, values=stats_view.cluster_size, name="Cluster Size", hole=0.4))
strings_pie = go.Figure(
    go.Pie(labels=stats_view.prediction, values=stats_view.unique_strings, name="Unique Strings", hole=0.4))
patterns_pie = go.Figure(
    go.Pie(labels=stats_view.prediction, values=stats_view.unique_patterns, name="Unique Patterns", hole=0.4))

date_analysis = fig['data'][0]['x'][0].date()

def makeTable(table):
    table.columns = [col.replace('_', ' ').title() for col in table.columns]
    return dbc.Table.from_dataframe(table, id='stats-view', striped=True, bordered=True, hover=True)


def makeDonut(fig, vertical_height_pct=45):
    name = fig['data'].__getitem__(0).name
    return dcc.Graph(
        id=f'{name.lower().replace(" ", "-")}-pie',
        figure=fig.update_layout(
            title={'text': name, 'x': 0.5, 'xanchor': 'center',
                   'font': dict(family="Cambria", size=20, )},
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)'
        ), style={'height': f'{vertical_height_pct}vh'}
    )


def makeTimePlot(fig, vertical_height_pct=50):
    return dcc.Graph(
        id='time-plot',
        figure=fig.update_layout(
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)'
        ), style={'height': f'{vertical_height_pct}vh'}
    )


app = dash.Dash("FTS Failures", external_stylesheets=[dbc.themes.LITERA])
# app = JupyterDash(external_stylesheets=[dbc.themes.LITERA])

app.title = 'FTS Failures'
app.layout = html.Div([
    # Headings
    dbc.Card([
        dbc.Row([
            dbc.Col(
                html.H3(
                    f'FTS Failures on {date_analysis}',
                    className='d-flex justify-content-center',
                    style={'font-family': 'cambria math', 'margin': '1%'}),
                width={'size': 9, 'offset': 1}),
            dbc.Col(
                dcc.Markdown('''for support contact: [*luca.clissa2@unibo.it*](mailto:luca.clissa2@unibo.it)''',
                             style={'font-size': '12px', 'margin-top': '5%', 'margin-right': '0px'}),
                width={'size': 2})
        ])
    ], style={"height": "8vh"}),

    # First row
    dbc.Card([
        dbc.Row([
            # Time evolution plot
            dbc.Col(
                makeTimePlot(fig),
                width={'size': 6}),
            # Numeric stats table
            dbc.Col(
                makeTable(stats_view),
                width={"size": 4, 'offset': 1}, align='center'
            )
        ])
    ], style={'height': '50vh'}),

    # Second row: pie charts for cluster size, unique strings and unique patterns
    dbc.Card([
        dbc.Row([
            dbc.Col(
                makeDonut(size_pie)
            ),
            dbc.Col(
                makeDonut(strings_pie)
            ),
            dbc.Col(
                makeDonut(patterns_pie)
            )
        ])
    ])

], style={"height": "100vh"})

# Run app and display result inline in the notebook
if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
    import plotly.io as pio

    pio.write_html(app, file='/home/luca/Downloads/dashboard.html', auto_open=True)
