
import os
import numpy as np
import pandas as pd

import plotly.graph_objs as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import base64
import io

class EEGDataloader:
    def __init__(self):
       pass
    def extractTimeValuefromtxt(self,txt):
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

    def valueConversion(self,data):
        data = (data - 2048) * 100 /2048
        return data

    def load_txt(self,file_path):
        txt= open(file_path, 'r')
        data, time= self.extractTimeValuefromtxt(txt)
        data= self.valueConversion(data)
        return data,time
    
    def load_txt_by_buttom(self, contents):
        _, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        data, time= self.extractTimeValuefromtxt(io.StringIO(decoded.decode('utf-8')))
        data= self.valueConversion(data)
        return data, time
    
    
class Annotation:
    
    def __init__(self, sf, patient):
      
        self.sf= sf
        self.patient= patient
        
    def create_dataframe(self, time_ch1, start_pos, end_pos):
        start_time = time_ch1[start_pos]
        end_time = time_ch1[end_pos]
        duration_in_ms = int((end_pos - start_pos)/self.sf * 1000)
        
        annotation_data = {
            'patient': [self.patient],
            'start_idx': [start_pos],
            'start_time': [start_time],
            'end_idx': [end_pos],
            'end_time': [end_time],
            'duration_ms': [duration_in_ms]
        }

        return pd.DataFrame(annotation_data, index=[0])
    
    def display(self,start_pos, start_time, end_pos, end_time, duration_in_ms):
    
        return (
            f'start pos/time: {start_pos:.1f}/{start_time}\n'
            f'end pos/time: {end_pos:.1f}/{end_time}\n'
            f'duration: {duration_in_ms} ms'
        )
    def record_or_delete(self, current_annotation, annot_export):
    
        # annot_export is empty then add annot directly
        if annot_export.shape[0]==0:

            annot_export= annot_export.append(current_annotation,ignore_index=True)
            print('draw annotation')
            print(annot_export.shape)
        else:
            # check if annot in annot_export, if in the annot_export means it's erasing the annot
            is_in_pd= annot_export.isin(current_annotation.to_dict(orient='list')).all(axis=1)
            print(is_in_pd)
            if is_in_pd.any():
                matching_row_idx= annot_export.index[is_in_pd].item()
                annot_export=annot_export[is_in_pd]
                print(f"Erase row {matching_row_idx} annotation ")
                print(annot_export.shape)
            else:
                annot_export= annot_export.append(current_annotation,ignore_index=True)
                print('draw annotation')
                print(annot_export.shape)
        return annot_export
    
#folder_path= 'c:/Users/annie/Desktop/OHCA_data/P10'
# reader= EEGDataloader()
# folder_path= "J:/我的雲端硬碟/thesis/OHCA_data/P10"
# file_path= os.path.join(folder_path, 'EEG1_nonseizure1.txt')
# data_ch1, time_ch1= reader.load_txt(file_path)
# file_path= os.path.join(folder_path, 'EEG2_nonseizure1.txt')
# data_ch2, time_ch2= reader.load_txt(file_path)
# assert data_ch1.shape== data_ch2.shape

patient=10
sf=128
# x_values_row = np.arange(0,len(data_ch1),1)
# y_values_row_ch1 = data_ch1
# y_values_row_ch2 = data_ch2
# print(y_values_row_ch1.shape)

eeg_loader= EEGDataloader()
annot_export= pd.DataFrame(columns=['patient', 'start_idx', 'start_time', 'end_idx','end_time','duration_ms'])
annotation = Annotation(sf, patient)

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
    html.P('height:'),
    dcc.Input(id='input-height', type='number', placeholder='Enter height', value=598),
    html.P('pixelsperinch:'),
    dcc.Input(id='input-pixelsperinch', type='number', placeholder='Enter pixelsperinch', value=105),
    
    html.P('4. Upload   :'),
    html.P('EEG ch1 txt \n '),
    dcc.Upload(
    id='upload-ch1',
    children=html.Button('Upload a file'),
    multiple=False
    ),
   
    html.P('EEG ch2 txt  \n '),
    dcc.Upload(
    id='upload-ch2',
    children=html.Button('Upload a file'),
    multiple=False
    ),
    
    html.Button('Enter', id='update-button'),
    
    dcc.Graph(id='plot-container',
              figure={},
              config={'modeBarButtonsToAdd':['drawrect','eraseshape']}),
    #html.Div(id='plot-container')
    html.Button('Export annotations', id='export-annotations-button'),
    dcc.Download(id='export-annot-csv'),
    html.Pre(id='annotation-info', children='annotation info\n'),
    
    html.Div(id='ch1-data', style={'display': 'none'}),
    html.Div(id='ch1-time', style={'display': 'none'}),
    html.Div(id='ch2-data', style={'display': 'none'}),
    html.Div(id='ch2-time', style={'display': 'none'})
])
@app.callback(
    [Output('ch1-data', 'children'),
     Output('ch1-time', 'children')],
    [Input('upload-ch1', 'contents')]
)
def load_ch1(contents):
    if contents is None:
        raise PreventUpdate
    reader= EEGDataloader()
    return reader.load_txt_by_buttom(contents)

@app.callback(
    [Output('ch2-data', 'children'),
     Output('ch2-time', 'children')],
    [Input('upload-ch2', 'contents')]
)
def load_ch2(contents):
    if contents is None:
        raise PreventUpdate
    reader= EEGDataloader()
    return reader.load_txt_by_buttom(contents)
# Define callback to update graph
@app.callback(
    Output('plot-container', 'figure'),
    [Input('update-button', 'n_clicks'),
     Input('ch1-data', 'children'),
     Input('ch1-time', 'children'),
     Input('ch2-data', 'children'),
     Input('ch2-time', 'children')],
    [State('input-width', 'value'),
     State('input-height', 'value'),
     State('input-pixelsperinch', 'value')]
)
def update_graph(n_clicks,data_ch1, time_ch1, data_ch2, time_ch2, \
                 width,height, pixelsperinch):
    if n_clicks is None:
        raise PreventUpdate
    
    assert len(data_ch1)== len(data_ch2)
    x_values_row = np.arange(0,len(data_ch1),1)
    fig= make_subplots(rows=2, cols=1,shared_xaxes=True,subplot_titles=[f'F3-P3', f'F4-P4'])
    fig.add_trace(go.Scatter(x=x_values_row, y=data_ch1, mode='lines'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_values_row, y=data_ch2, mode='lines'), row=2, col=1)

    # adjust plot based on monitor
    x_start = 0
    x_timebase , y_base= 30, 7  # mm/sec, mV/mm
    width_mm = width/ (pixelsperinch / 2.54)*10 # pixel to mm
    height_mm = height/ (pixelsperinch / 2.54)*10 # pixel to mm
    x_end = int((sf-8)* ((width_mm / x_timebase)))
    initial_x_range= [x_values_row[x_start], x_values_row[x_end - 1]]
    initial_y_range= [-height_mm*0.22875817*y_base/2, height_mm*0.22875817*y_base/2]
    
    # set ticks as string time
    tick_interval = 200
    tick_positions = np.arange(0, len(x_values_row), tick_interval)
    tick_labels = time_ch1[::tick_interval]
    
    
    fig.update_layout(
        title=f'Patient {patient}',
        title_x= 0.5,
        width= width+20,
        showlegend=False,
        margin=dict(l=10, r=10, b=0, t=40),
        dragmode='drawrect', # define dragmode
        newshape=dict(line_color='cyan')
    
    )
    fig.update_xaxes(range=initial_x_range,tickvals=tick_positions, ticktext=tick_labels ,row=1, col=1)
    fig.update_xaxes(range=initial_x_range,tickvals=tick_positions, ticktext=tick_labels, row=2, col=1)
    fig.update_yaxes(range=initial_y_range, row=1, col=1, fixedrange=True)
    fig.update_yaxes(range=initial_y_range, row=2, col=1, fixedrange=True)

   

    return fig

@app.callback(
    Output('annotation-info', 'children'),
    [Input('plot-container', 'relayoutData'),
     Input('ch1-time', 'children')],
    [State('annotation-info', 'children')])

def rect_annotation_added(fig_data, time_ch1, content):
    global annot_export
    if fig_data is None:
        return dash.no_update
    
    if 'shapes' in fig_data :
       
        # when only annotation is deleted shape is empty []
        if len(fig_data['shapes'])==0 and len(annot_export)==1:
            annot_export= pd.DataFrame(columns=['patient', 'start_idx', 'start_time', 'end_idx','end_time','duration_ms'])
            print('Erase the only annotation')
            print('annotation is empty now')
        else:
            line = fig_data['shapes'][-1]
            start_pos= int(line['x0'])
            end_pos= int(line['x1'])
            current_annotation = annotation.create_dataframe(time_ch1, start_pos, end_pos)
            content= ""
            content= annotation.display(
                start_pos, current_annotation['start_time'].item(),
                end_pos, current_annotation['end_time'].item(),
                current_annotation['duration_ms'].item()
            )
            annot_export= annotation.record_or_delete(current_annotation, annot_export)
                
    return content

@app.callback(
    Output("export-annot-csv", "data"),
    [Input("export-annotations-button", "n_clicks"),
     Input('ch1-time', 'children')],
    prevent_initial_call=True,
)
def export_csv(n_clicks, time_ch1):
    if n_clicks is None:
        raise PreventUpdate
    startime= time_ch1[0].replace(':', '')
    endtime  = time_ch1[-1].replace(':', '')
    return dcc.send_data_frame(annot_export.to_csv,f'P{patient}_{startime}_{endtime}_annotation.csv')


if __name__ == '__main__':
    app.run_server(debug=True)



