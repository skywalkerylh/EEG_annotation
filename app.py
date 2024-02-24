
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
    def extract_patient_number_from_filename(self,filename):
        parts = filename.split('_')
        patient_number_part = parts[0].replace('P', '')
       
        return patient_number_part
    
class Annotation:
    
    def __init__(self, sf):
        self.sf= sf

    def create_dataframe(self, patient, time_ch1, start_pos, end_pos):
        start_time = time_ch1[start_pos]
        end_time = time_ch1[end_pos]
        duration_in_ms = int((end_pos - start_pos)/self.sf * 1000)
        
        annotation_data = {
            'patient': [patient],
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
            annot_export = pd.concat([annot_export, current_annotation], ignore_index=True)
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
                annot_export = pd.concat([annot_export, current_annotation], ignore_index=True)
                print('draw annotation')
                print(annot_export.shape)
        return annot_export

#patient=10
sf=128
eeg_loader= EEGDataloader()
annot_export= pd.DataFrame(columns=['patient', 'start_idx', 'start_time', 'end_idx','end_time','duration_ms'])
annotation = Annotation(sf)

# web
app = dash.Dash(__name__)
app.layout = html.Div([
    html.P('Upload   :'),
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
  
    html.Button('Export annotations', id='export-annotations-button'),
    dcc.Download(id='export-annot-csv'),
    html.Pre(id='annotation-info', children='annotation info\n'),
    
    html.Div(id='ch1-data', style={'display': 'none'}),
    html.Div(id='ch1-time', style={'display': 'none'}),
    html.Div(id='ch2-data', style={'display': 'none'}),
    html.Div(id='ch2-time', style={'display': 'none'}),
    html.Div(id='patient', style={'display': 'none'})
])
# Define callback to update graph
@app.callback(
    [Output('plot-container', 'figure'),
     Output('ch1-time', 'children'),
     Output('patient', 'children')],
    [Input('update-button', 'n_clicks'),
     Input('ch1-data', 'children'),
     Input('ch1-time', 'children'),
     Input('ch2-data', 'children'),
     Input('ch2-time', 'children'),
     Input('patient', 'children')],
    [State('upload-ch1', 'contents'),
     State('upload-ch2', 'contents'),
     State('upload-ch1', 'filename')]
)
def update_graph(n_clicks,data_ch1, time_ch1, data_ch2, time_ch2, patient, \
                 #width,height, pixelsperinch, \
                 txt_ch1, txt_ch2, filename):
    if n_clicks is None:
        raise PreventUpdate
    #load data
    reader= EEGDataloader()
    data_ch1, time_ch1= reader.load_txt_by_buttom(txt_ch1)
    data_ch2, time_ch2= reader.load_txt_by_buttom(txt_ch2)
    assert len(data_ch1)== len(data_ch2)
    patient= reader.extract_patient_number_from_filename(filename)
    
    #plot fig
    x_values_row = np.arange(0,len(data_ch1),1)
    fig= make_subplots(rows=2, cols=1,shared_xaxes=True,subplot_titles=[f'F3-P3', f'F4-P4'])
    fig.add_trace(go.Scatter(x=x_values_row, y=data_ch1, mode='lines'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_values_row, y=data_ch2, mode='lines'), row=2, col=1)

    # adjust plot based on monitor
    x_start = 0
    x_end= sf*10
    initial_x_range= [x_values_row[x_start], x_values_row[x_end - 1]]
    initial_y_range= [min(data_ch1), max(data_ch1)]
    
    # set ticks as string time
    tick_interval = sf
    tick_positions = np.arange(0, len(x_values_row), tick_interval)
    tick_labels = time_ch1[::tick_interval]
    
    
    fig.update_layout(
        title=f'Patient {patient}',
        title_x= 0.5,
        showlegend=False,
        margin=dict(l=10, r=10, b=0, t=40),
        dragmode='drawrect', # define dragmode
        newshape=dict(line_color='cyan')
    
    )
    fig.update_xaxes(range=initial_x_range,tickvals=tick_positions, ticktext=tick_labels ,row=1, col=1, dtick=128)
    fig.update_xaxes(range=initial_x_range,tickvals=tick_positions, ticktext=tick_labels, row=2, col=1, dtick=128)
    fig.update_yaxes(range=initial_y_range, row=1, col=1, fixedrange=True, dtick=50)
    fig.update_yaxes(range=initial_y_range, row=2, col=1, fixedrange=True, dtick=50)

   

    return fig, time_ch1, patient

@app.callback(
    Output('annotation-info', 'children'),
    [Input('plot-container', 'relayoutData'),
     Input('ch1-time', 'children')],
    [State('annotation-info', 'children'),
     State('patient', 'children')])

def rect_annotation_added(fig_data, time_ch1, content, patient):
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
            current_annotation = annotation.create_dataframe(patient, time_ch1, start_pos, end_pos)
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
    [State('patient', 'children')],
    prevent_initial_call=True,
)
def export_csv(n_clicks, time_ch1, patient):
    if n_clicks is None:
        raise PreventUpdate
    startime= time_ch1[0].replace(':', '')
    endtime  = time_ch1[-1].replace(':', '')
    return dcc.send_data_frame(annot_export.to_csv,f'P{patient}_{startime}_{endtime}_annotation.csv')


if __name__ == '__main__':
    app.run_server(debug=False, host= '0.0.0.0',port=8080)



