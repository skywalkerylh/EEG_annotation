
import os
import numpy as np
import pandas as pd

import plotly.graph_objs as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px

def extractTimeValuefromtxt(txt):
    '''
    return
    values: int array
    time : string list
    '''
    data = txt.readlines()
    time= [line.split('\t')[0]for line in data]
    values= [line.split('\t')[1].split('\n')[0] for line in data]
    values= [int(line) for line in values]
    values= np.array(values)
    return values, time

def valueConversion(data):
    data = (data - 2048) * 100 /2048
    return data

def readEEGtxt(file_path):
    txt= open(file_path, 'r')
    data, time= extractTimeValuefromtxt(txt)
    data= valueConversion(data)
    return data,time

#folder_path= 'c:/Users/annie/Desktop/OHCA_data/P10'
folder_path= "J:/我的雲端硬碟/thesis/OHCA_data/P10"
file_path= os.path.join(folder_path, 'EEG1_event1.txt')
data_ch1, time_ch1= readEEGtxt(file_path)
file_path= os.path.join(folder_path, 'EEG2_event1.txt')
data_ch2, time_ch2= readEEGtxt(file_path)
assert data_ch1.shape== data_ch2.shape

patient=10
sf=128

# web
app = dash.Dash(__name__)
app.layout = html.Div([
    html.P('1. check browser resolution(width*height):'),
    dcc.Link('Click here', href='https://mdigi.tools/browser-resolution/', target='_blank'),
    html.Br(),
    html.P('2. check pixel per inch :'),
    dcc.Link('Click here', href='https://dpi.lv/', target='_blank'),
    html.Br(),
    
    html.P('3. Input  :'),
    html.P('width:'),
    dcc.Input(id='input-width', type='number', placeholder='Enter width', value=1257),
    html.P('pixelsperinch:'),
    dcc.Input(id='input-pixelsperinch', type='number', placeholder='Enter pixelsperinch', value=105),
   
    html.Button('Enter', id='update-button'),
    dcc.Graph(id='plot-container', figure={})
    #html.Div(id='plot-container')
    
])

# Define callback to update graph
@app.callback(
    Output('plot-container', 'figure'),
    [Input('update-button', 'n_clicks')],
    [dash.dependencies.State('input-width', 'value'),
     dash.dependencies.State('input-pixelsperinch', 'value')]
)
def update_graph(n_clicks, width, pixelsperinch):
    if n_clicks is None:
        raise PreventUpdate

    x_values_row = np.arange(0,len(data_ch1),1)
    y_values_row_ch1 = data_ch1
    y_values_row_ch2 = data_ch2
    print(y_values_row_ch1.shape)

    fig_row = make_subplots(rows=2, cols=1,shared_xaxes=True,subplot_titles=[f'F3-P3', f'F4-P4'])
    fig_row.add_trace(go.Scatter(x=x_values_row, y=y_values_row_ch1, mode='lines'), row=1, col=1)
    fig_row.add_trace(go.Scatter(x=x_values_row, y=y_values_row_ch2, mode='lines'), row=2, col=1)

    x_start = 0
    timebase = 30  # mm/sec
    width_mm = width/ (pixelsperinch / 2.54)*10 # pixel to mm
    x_end = int((sf-8)* ((width_mm / timebase)))
    initial_x_range= [x_values_row[x_start], x_values_row[x_end - 1]]
    initial_y_range= [min(data_ch1),max(data_ch1)]
    
    # set ticks as string time
    tick_interval = 200
    tick_positions = np.arange(0, len(x_values_row), tick_interval)
    tick_labels = time_ch1[::tick_interval]
    
    fig_row.update_layout(
        title=f'Patient {patient}',
        title_x= 0.5,
        width= width+20,
        showlegend=False,
        margin=dict(l=10, r=10, b=0, t=40)
       
    )
    fig_row.update_xaxes(range=initial_x_range,tickvals=tick_positions, ticktext=tick_labels ,row=1, col=1)
    fig_row.update_xaxes(range=initial_x_range,tickvals=tick_positions, ticktext=tick_labels, row=2, col=1)
    fig_row.update_yaxes(range=initial_y_range, row=1, col=1, fixedrange=True)
    fig_row.update_yaxes(range=initial_y_range, row=2, col=1, fixedrange=True)


    
    return fig_row


if __name__ == '__main__':
    app.run_server(debug=True)



