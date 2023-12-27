import datetime

import dash
import numpy as np
import plotly
import plotly.subplots
from dash import dcc, html
from dash.dependencies import Input, Output

# pip install pyorbital
# satellite = Orbital('TERRA')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(
    html.Div([
        html.H4('TERRA Satellite Live Feed'),
        html.Div(id='live-update-text'),
        dcc.Graph(id='live-update-graph'),
        dcc.Interval(
            id='interval-component',
            interval=1 * 100,  # in milliseconds
            n_intervals=0
        )
    ])
)

lon = [0]
lat = [0]
alt = [0]
times = [datetime.datetime.now()]


@app.callback(Output('live-update-text', 'children'),
              Input('interval-component', 'n_intervals'))
def update_metrics(n):
    # lon, lat, alt = satellite.get_lonlatalt(datetime.datetime.now())
    global lon
    global alt
    global lat
    global times
    times.append(datetime.datetime.now())
    lon.append(lon[-1] + np.random.rand())
    lat.append(lat[-1] + np.random.rand())
    alt.append(alt[-1] + np.random.rand())
    style = {'padding': '5px', 'fontSize': '16px'}
    return [
        html.Span('Longitude: {0:.2f}'.format(lon[-1]), style=style),
        html.Span('Latitude: {0:.2f}'.format(lat[-1]), style=style),
        html.Span('Altitude: {0:0.2f}'.format(alt[-1]), style=style)
    ]


# Multiple components can update everytime interval gets fired.
@app.callback(Output('live-update-graph', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_graph_live(n):
    # satellite = Orbital('TERRA')
    data = {
        'time': [],
        'Latitude': [],
        'Longitude': [],
        'Altitude': []
    }

    print(data)
    # Collect some data
    for i in range(len(times)):
        # time = datetime.datetime.now() - datetime.timedelta(seconds=i*20)
        # lon, lat, alt = satellite.get_lonlatalt(
        #     time
        # )
        data['Longitude'].append(lon[i])
        data['Latitude'].append(lat[i])
        data['Altitude'].append(alt[i])
        data['time'].append(times[i])

    print("Before create fig")
    # Create the graph with subplots
    # fig = plotly.subplots.make_subplots(rows=2, cols=1, vertical_spacing=0.2)
    fig = plotly.subplots.make_subplots(rows=2, cols=1)
    print("After create fig")
    fig['layout']['margin'] = {
        'l': 30, 'r': 10, 'b': 30, 't': 10
    }
    fig['layout']['legend'] = {'x': 0, 'y': 1, 'xanchor': 'left'}
    print("After figure")
    print(data)
    fig.add_trace({
        'x': data['time'],
        'y': data['Altitude'],
        'name': 'Altitude',
        'mode': 'lines+markers',
        'type': 'scatter'
    }, 1, 1)
    fig.add_trace({
        'x': data['Longitude'],
        'y': data['Latitude'],
        'text': data['time'],
        'name': 'Longitude vs Latitude',
        'mode': 'lines+markers',
        'type': 'scatter'
    }, 2, 1)

    return fig


if __name__ == '__main__':
    app.run_server(debug=False)
