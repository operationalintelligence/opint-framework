# -*- coding: utf-8 -*-
import datetime
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


def makeTable(table):
    table.columns = [col.replace('_', ' ').title() for col in table.columns]
    return dbc.Table.from_dataframe(table, id='stats-view', striped=True, bordered=True, hover=True)


def makeDonut(fig, vertical_height_pct=40):
    name = fig['data'].__getitem__(0).name
    return dcc.Graph(
        id=f'{name.lower().replace(" ", "-")}-pie',
        figure=fig.update_layout(
            title=name,
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
                html.H1(
                    f'FTS Failures on {datetime.datetime.now().date()}',
                    style={'align': 'center'}),
                width={'size': 8, 'offset': 1}),
            dbc.Col(
                dcc.Markdown('''###### for support contact [*luca.clissa2@unibo.it*][mailto:luca.clissa2@unibo.it]'''),
                width={'size': 2, 'offset': 1, 'order': 'last'})
        ])
    ], style={"height": "10vh"}),

    # First row
    dbc.Card([
        dbc.Row([
            dbc.Col(
                makeTimePlot(fig, 50),
                width={'size': 6}),
            dbc.Col(
                makeTable(stats_view),
                # dash_table.DataTable(
                #         id='stats-view',
                #         columns=[{"name": i.replace("_", " ").title(), "id": i} for i in stats_view.columns],
                #         data=stats_view.to_dict('records'),
                #     ),
                width={"size": 4, 'offset': 1}, align='center'
            )
        ])
    ], style={'height': '50vh'}),

    # Second row
    dbc.Card([
        dbc.Row([
            dbc.Col(
                makeDonut(size_pie, 45)
            ),
            dbc.Col(
                makeDonut(strings_pie, 45)
            ),
            dbc.Col(
                makeDonut(patterns_pie, 45)
            )
        ])
    ])

], style={"height": "100vh"})

# Run app and display result inline in the notebook
if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
    import plotly.io as pio

    pio.write_html(app, file='/home/luca/Downloads/dashboard.html', auto_open=True)
