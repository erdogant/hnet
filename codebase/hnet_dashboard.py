#https://dash.plot.ly/dash-core-components
#%% Libraries
import os
import base64
from pathlib import Path
# popular libraries
import numpy as np
import pandas as pd
#import Core
# Front-end
import dash_dangerously_set_inner_html
import codecs
import dash
#import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import webbrowser
# Method
import hnet as hnet
import helpers.picklefast as picklefast

global labels

#%% Initializatoin
TMP_DIRECTORY     = './tmp/'
HNET_DIR_STABLE   = './results/stable/'
HNET_DIR_TMP      = './results/tmp/'
BACKGROUND_IMAGE  = 'url(./static/background.jpg)'
TMP_DIRECTORY_DEL = False

# Create directories
path=Path(TMP_DIRECTORY)
path.mkdir(parents=True, exist_ok=True)
path=Path(HNET_DIR_TMP)
path.mkdir(parents=True, exist_ok=True)
path=Path(HNET_DIR_STABLE)
path.mkdir(parents=True, exist_ok=True)

# At initialization remove content in the tmp directory
if TMP_DIRECTORY_DEL:
    print('[HNET-GUI] Cleaning files from tmp directory..')
    remfiles=os.listdir(TMP_DIRECTORY)
    for remfile in remfiles:
        if os.path.isfile(remfile): os.remove(os.path.join(TMP_DIRECTORY,remfile))

# Extract HNet results from tmp and stable directories
HNET_PATH_STABLE = [{'label':i,'value':os.path.join(HNET_DIR_STABLE,i)} for i in os.listdir(HNET_DIR_STABLE)]
HNET_PATH_TMP    = [{'label':i,'value':os.path.join(HNET_DIR_TMP,i)} for i in os.listdir(HNET_DIR_TMP)]
HNET_PATH_TOTAL  = HNET_PATH_STABLE + HNET_PATH_TMP

#%%
#df=pd.read_csv('D://stack/TOOLBOX_PY/DATA/OTHER/titanic/titanic_train.csv')
#labels=[{'label':i,'value':i} for i in df.columns.unique()]
#if 'labels' not in globals():
#    labels=[{'label':'','value':''}]
#    print('setup labels first time')

#%%
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app=dash.Dash(__name__, external_stylesheets=external_stylesheets)
#app.css.append_css({'external_url':external_stylesheets}) # Required for making columns
app.scripts.config.serve_locally = True
app.scripts.append_script({"external_url": ['https://d3js.org/d3.v3.min.js']})
# Normally Dahs creates its own Flas server internally but by creating our own, we cancreate a route for downloading files
# server=Flask(__name__)
#app=dash.Dash(server=server)
#app.css.append_css({'external_url':'./static/bWLwgP.css'}) # Required for making columns

#%% Styles
#STYLE_UNDERSCRIPT={"max-width":"90%","textAlign":"center","color":"white","font-size":"20px"}
#STYLE_BACKGROUND={"background-image":"BACKGROUND_IMAGE","background-repeat":"no-repeat","background-size":"cover","background-position":"center"}
#DRAG_AND_DROP={
#        "width":"90%",
#        "height":"320px",
#        "lineHeight":"60px",
#        "borderWidth":"10px",
#        "borderStyle":"dashed",
#        "borderRadius":"5px",
#        "margin":"10px",
#        "backgroundColor":"",
#        "color":"white",
#        "font-size":"22px",
#        "display":"inline-block",
#        }

#%% Setup webpage

GUIelements = html.Div([
        html.Div([html.H5("HNets: Graphical Hypergeometric Networks")], style={'textAlign':'left','width':'100%','backgroundColor':'#e0e0e0'}),

        html.Div([

            html.Div([
                html.H6('Parameters'),
                dcc.Input(id='k-id', placeholder='Enter k..', type='text', value=1, style={"width": "100%"}), 
                #dcc.Input(id='alpha-id', placeholder='Enter alpha..', type='text', value=0.05, style={"width": "100%"}), 
                #dcc.Input(id='ymin-id', placeholder='Enter a value for y_min..', type='text', value=10, style={"width": "100%"}), 
                #dcc.Checklist(id='checkbox-id', options=[{'label':'drop nan','value':'True'}], value=['True'], style={"width": "100%"}),
                dcc.Input(id='perc_min_num-id', placeholder='Minimum percentage..', type='text', value='', style={"width": "100%"}), 
                dcc.Input(id='excl_background-id', placeholder='Remove background..', type='text', value='', style={"width": "100%"}), 

                dcc.Dropdown(id='alpha-id',
                    options=[
                        {'label': '0.0001', 'value': '0.0001'},
                        {'label': '0.001', 'value': '0.001'},
                        {'label': '0.01', 'value': '0.01'},
                        {'label': '0.05', 'value': '0.05'},
                        {'label': '0.1', 'value': '0.1'},
                        {'label': '1', 'value': '1'},
                    ],
                    value='0.05', style={"width": "100%"}),

                dcc.Dropdown(id='ymin-id',
                    options=[
                        {'label': '1', 'value': '1'},
                        {'label': '5', 'value': '5'},
                        {'label': '10', 'value': '10'},
                        {'label': '20', 'value': '20'},
                        {'label': '50', 'value': '50'},
                        {'label': '100', 'value': '100'},
                    ],
                    value='10', style={"width": "100%"}),

                dcc.Dropdown(id='specificity-id',
                    options=[
                        {'label': 'Low', 'value': 'low'},
                        {'label': 'Medium', 'value': 'medium'},
                        {'label': 'High', 'value': 'high'}
                    ],
                    value='medium', style={"width": "100%"}),
                
                dcc.Dropdown(id='multtest-id',
                    options=[
                        {'label': 'Holm', 'value': 'holm'},
                        {'label': 'Bonferroni', 'value': 'bonferroni'},
                        {'label': 'Hommel', 'value': 'hommel'},
                        {'label': 'Benjamini/Hochberg', 'value': 'fdr_bh'},
                        {'label': 'Benjamini/Yekutieli', 'value': 'fdr_by'},
                        {'label': 'Sidak', 'value': 'sidak'},
                        {'label': 'Holm-sidak', 'value': 'holm-sidak'},
                    ],
                    value='holm', style={"width": "100%"}),


                html.Div(id="OUTPUT_CSV"),

                dcc.Upload(id="UPLOAD_BOX",children=html.Div(["Drag and drop or click to select a file to upload."]),
                       style={
                        #"width": "100%",
                        "height": "250px",
                        "lineHeight": "60px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "5px",
                        "textAlign": "center",
                        "margin": "10px",
                        'backgroundColor':'white',
                    },
                    multiple=False), 
               
                # Create drop-down for 
                dcc.Dropdown(id='results-id', options=[{'label':i,'value':os.path.join(HNET_DIR_STABLE,i)} for i in os.listdir(HNET_DIR_STABLE)], value='', style={"width": "100%"}),
                html.Div(id="results-output")


            ], className="six columns", style={"width": "15%", "border":"1px black solid", "height": "700px",'backgroundColor':''}),

            # --------------------- NETWORK plotly  ------------------------- # 
            html.Div(dash_dangerously_set_inner_html.DangerouslySetInnerHTML(open('D://stack/TOOLBOX_PY/PROJECTS/HNET/codebase/assets/index.html','r').read())),
            # --------------------------------------------------------------- # 

#            html.Div(id="results-output", className="six columns", style={"width": "80%", "border":"1px black solid", "height": "700px"}),
#            html.Iframe('D://stack/TOOLBOX_PY/PROJECTS/HNET/codebase/results/stable/sprinkler_data_1000_10_1_holm_medium_None_None/index.html'),
#            html.Iframe(src=app.get_asset_url('D://stack/TOOLBOX_PY/PROJECTS/HNET/codebase/assets/index.html')),
#            html.Div(html.Iframe(src='D://stack/TOOLBOX_PY/PROJECTS/HNET/codebase/assets/index.html'), className="six columns"),

#            https://github.com/plotly/dash/issues/71
#            https://dash.plot.ly/external-resources
#            https://stackoverflow.com/questions/52013320/how-can-i-add-raw-html-javascript-to-a-dash-application

            # Dit werkt bijna
#            html.Div(dash_dangerously_set_inner_html.DangerouslySetInnerHTML(codecs.open('D://stack/TOOLBOX_PY/PROJECTS/HNET/codebase/assets/index.html', 'r', 'utf-8').read())),
            html.Div(dash_dangerously_set_inner_html.DangerouslySetInnerHTML(open('D://stack/TOOLBOX_PY/PROJECTS/HNET/codebase/assets/index.html','r').read())),

#            html.Div(dash_dangerously_set_inner_html.DangerouslySetInnerHTML('''
#            <head>
#            <script type="text/javascript" src="D://stack/TOOLBOX_PY/PROJECTS/HNET/codebase/assets/d3.v3.js"></script>
#            <link rel="stylesheet" href="D://stack/TOOLBOX_PY/PROJECTS/HNET/codebase/assets/style.css">
#            </head>
#            <body>
#            <script type="application/json" id="d3graph"> {
#                "links": [
#                    {
#                        "weight": 54.82158881980537,
#                        "edge_weight": 54.82158881980537,
#                        "edge_width": 12.247324856961392,
#                        "source_label": "Sprinkler_1",
#                        "target_label": "Cloudy_0",
#                        "source": 3,
#                        "target": 0
#                    },
#                    {
#                        "weight": 22.833715739709238,
#                        "edge_weight": 22.833715739709238,
#                        "edge_width": 5.684618307217716,
#                        "source_label": "Rain_0",
#                        "target_label": "Sprinkler_1",
#                        "source": 4,
#                        "target": 3
#                    },
#                    {
#                        "weight": 24.53340465164335,
#                        "edge_weight": 24.53340465164335,
#                        "edge_width": 6.0333304434371575,
#                        "source_label": "Wet_Grass_1",
#                        "target_label": "Sprinkler_1",
#                        "source": 6,
#                        "target": 3
#                    },
#                    {
#                        "weight": 54.82158881980537,
#                        "edge_weight": 54.82158881980537,
#                        "edge_width": 12.247324856961392,
#                        "source_label": "Sprinkler_0",
#                        "target_label": "Cloudy_1",
#                        "source": 2,
#                        "target": 1
#                    },
#                    {
#                        "weight": 92.60959390993409,
#                        "edge_weight": 92.60959390993409,
#                        "edge_width": 20.0,
#                        "source_label": "Rain_1",
#                        "target_label": "Cloudy_1",
#                        "source": 5,
#                        "target": 1
#                    },
#                    {
#                        "weight": 10.636268384000315,
#                        "edge_weight": 10.636268384000315,
#                        "edge_width": 3.182161596481509,
#                        "source_label": "Wet_Grass_1",
#                        "target_label": "Cloudy_1",
#                        "source": 6,
#                        "target": 1
#                    },
#                    {
#                        "weight": 22.833715739709238,
#                        "edge_weight": 22.833715739709238,
#                        "edge_width": 5.684618307217716,
#                        "source_label": "Rain_1",
#                        "target_label": "Sprinkler_0",
#                        "source": 5,
#                        "target": 2
#                    },
#                    {
#                        "weight": 67.01662541674791,
#                        "edge_weight": 67.01662541674791,
#                        "edge_width": 14.749286970813761,
#                        "source_label": "Wet_Grass_1",
#                        "target_label": "Rain_1",
#                        "source": 6,
#                        "target": 5
#                    }
#                ],
#                "nodes": [
#                    {
#                        "node_name": "Cloudy_0",
#                        "node_color": "#a6cee3",
#                        "node_size": "12",
#                        "node_size_edge": "0.10000000000000009",
#                        "node_color_edge": "#000000"
#                    },
#                    {
#                        "node_name": "Cloudy_1",
#                        "node_color": "#a6cee3",
#                        "node_size": "12",
#                        "node_size_edge": "0.10000000000000009",
#                        "node_color_edge": "#000000"
#                    },
#                    {
#                        "node_name": "Sprinkler_0",
#                        "node_color": "#b2df8a",
#                        "node_size": "15",
#                        "node_size_edge": "0.10000000000000009",
#                        "node_color_edge": "#000000"
#                    },
#                    {
#                        "node_name": "Sprinkler_1",
#                        "node_color": "#b2df8a",
#                        "node_size": "10",
#                        "node_size_edge": "0.10000000000000009",
#                        "node_color_edge": "#000000"
#                    },
#                    {
#                        "node_name": "Rain_0",
#                        "node_color": "#1f78b4",
#                        "node_size": "12",
#                        "node_size_edge": "0.10000000000000009",
#                        "node_color_edge": "#000000"
#                    },
#                    {
#                        "node_name": "Rain_1",
#                        "node_color": "#1f78b4",
#                        "node_size": "12",
#                        "node_size_edge": "0.10000000000000009",
#                        "node_color_edge": "#000000"
#                    },
#                    {
#                        "node_name": "Wet_Grass_1",
#                        "node_color": "#33a02c",
#                        "node_size": "14",
#                        "node_size_edge": "0.10000000000000009",
#                        "node_color_edge": "#000000"
#                    }
#                ]
#            }
#            </script>
#            <form>
#            <h3> Link threshold 0 <input type="range" id="thersholdSlider" name="points" value=9.0 min="9.0" max="93.0" onchange="threshold(this.value)"> 93.0 </h3>
#            </form>
#            <script  src="D://stack/TOOLBOX_PY/PROJECTS/HNET/codebase/assets/d3graphscript.js"></script>
#            </body>
#            </html>
#            ''')),



#            html.Div(dash_dangerously_set_inner_html.DangerouslySetInnerHTML('''<h1>Header</h1>'''),),
            # https://dash.plot.ly/external-resources
            # https://community.plot.ly/t/rendering-html-similar-to-markdown/6232/2

            
#            html.Div(html.Iframe(
#            # enable all sandbox features
#            # see https://developer.mozilla.org/en-US/docs/Web/HTML/Element/iframe
#            # this prevents javascript from running inside the iframe
#            # and other things security reasons
#            sandbox='',
#            srcDoc='''
#                <h3>IFrame</h3>
#                <script type="text/javascript">
#                    alert("This javascript will not be executed")
#                </script>
#            '''
#        ), className="six columns"),


#        html.Div(id='video-target'),
#            dcc.Dropdown(
#                id='video-dropdown',
#                options=[
#                    {'label': 'Video 1', 'value': 'video1'},
#                    {'label': 'Video 2', 'value': 'video2'},
#                    {'label': 'Video 3', 'value': 'video3'},
#                ],
#                value='video1'
#            )

            
        
        ], className="row", style={"width": "100%"}),
        

    ], className="row", style={"width": "100%"} #style={"max-width": "500px"},
)

#%% Run webpage
app.layout = html.Div([GUIelements])

#%%

#%%
#@app.callback(Output('video-target', 'children'), [Input('video-dropdown', 'value')])
#def embed_iframe(value):
#    videos = {
#        'video1': 'sea2K4AuPOk',
#        'video2': '5BAthiN0htc',
#        'video3': 'e4ti2fCpXMI',
#    }
##    https://community.plot.ly/t/how-to-load-html-file-directly-on-dash/8563
##    https://community.plot.ly/t/how-can-i-use-my-html-file-in-dash/7740/2
##    https://community.plot.ly/t/how-to-load-html-file-directly-on-dash/8563
#    
##    return html.Iframe(src=f'https://www.youtube.com/embed/{videos[value]}')
#    return dash_dangerously_set_inner_html.DangerouslySetInnerHTML('''<h1>Header</h1>''')
#    #return html.Iframe('D://stack/TOOLBOX_PY/PROJECTS/HNET/codebase/assets/index.html')
##    return(html.Iframe(
##        # enable all sandbox features
##        # see https://developer.mozilla.org/en-US/docs/Web/HTML/Element/iframe
##        # this prevents javascript from running inside the iframe
##        # and other things security reasons
##        sandbox='',
##        srcDoc='''
##            <h3>IFrame</h3>
##            <script type="text/javascript">
##                alert("This javascript will not be executed")
##            </script>
##        '''
##    ))
    
#%%
@app.callback(
    Output("results-output", "children"),
    [Input("results-id","value")],
)
def load_results(results_path):
    """Save uploaded files and regenerate the file list."""
    d3path=''
    #print(results_path)
#    d3path = os.path.join(HNET_DIR_STABLE,results_path,'index.html')
    if results_path!=None:
        # Set path
        if os.path.isfile(os.path.join(results_path,'index.html')):
            print(d3path)
            d3path=os.path.abspath(os.path.join(results_path,'index.html'))
            # open in webbrowser
            webbrowser.open(d3path, new=2)
            # https://plot.ly/python/network-graphs/
    # Return
    return(d3path)

#%%
@app.callback(
    Output("OUTPUT_CSV", "children"),
    [Input("UPLOAD_BOX","filename"), 
     Input("UPLOAD_BOX","contents"), 
     Input("ymin-id","value"), 
     Input("alpha-id","value"), 
     Input("k-id","value"),
     Input("excl_background-id","value"),
     Input("perc_min_num-id","value"),
     Input("specificity-id","value"),
     Input("multtest-id","value"),
     ],
)
def process_csv_file(uploaded_filenames, uploaded_file_contents, y_min, alpha, k, excl_background, perc_min_num, specificity, multtest):
    """Save uploaded files and regenerate the file list."""
    # Check input
    [args, runOK, runtxt]=check_input(uploaded_filenames, uploaded_file_contents, y_min, alpha, k, excl_background, perc_min_num, specificity, multtest)
    if runOK==False:
        for txt in runtxt: print('[HNET-GUI] %s' %txt)
        return(runtxt)

    print('alpha:%s' %args['alpha'])
    print('y_min:%s' %args['y_min'])
    print('k:%s' %args['k'])
    print('multtest:%s' %args['multtest'])
    print('excl_background:%s' %args['excl_background'])
    print('perc_min_num:%s' %args['perc_min_num'])
    print('specificity:%s' %args['specificity'])
    print('dropna:%s' %args['dropna'])
    print('File input: %s' %(args['uploaded_filenames']))
    
#    if uploaded_filenames is not None and uploaded_file_contents is not None:
    filepath        = save_file(args['uploaded_filenames'], args['uploaded_file_contents'], TMP_DIRECTORY)
    [_,filename, _] = splitpath(args['uploaded_filenames'])
    savepath        = os.path.join(HNET_DIR_STABLE,filename+'_'+str(args['y_min'])+'_'+str(args['k'])+'_'+str(args['multtest'])+'_'+str(args['specificity'])+'_'+str(args['perc_min_num'])+'_'+str(args['excl_background'])+'/')
    d3path          = os.path.join(savepath,'index.html')
    pklpath         = os.path.join(savepath,'hnet.pkl')

    print('Filepath %s' %(filepath))
    print('Savepath %s' %(savepath))
    print('d3path %s' %(d3path))
    print('pklpath %s' %(pklpath))
    
    # Make directory
    if not os.path.isdir(savepath):
        os.mkdir(savepath)
    # Make D3js path
    if not os.path.isfile(d3path):
        # Read file
        df = pd.read_csv(filepath)
        print(df.shape)
        labels=[{'label':i,'value':i} for i in df.columns.unique()]
        print(labels)

        out = hnet.main(df, alpha=args['alpha'], y_min=args['y_min'], k=args['k'], multtest=args['multtest'], dtypes='pandas', specificity=args['specificity'], perc_min_num=args['perc_min_num'], dropna=args['dropna'], excl_background=args['excl_background'], verbose=3)
        # Save pickle file
#            picklefast.save(os.path.abspath(pklpath), out)
        picklefast.save(pklpath, out)
        #G = hnet.plot_network(out, dist_between_nodes=0.4, scale=2)
        G = hnet.plot_d3graph(out, savepath=savepath, directed=False, showfig=False)
    else:
        print('dir exists, load stuff')
        out=picklefast.load(pklpath)
#            out=picklefast.load(os.path.abspath(pklpath))
        
    # Open in browser
    if os.path.isfile(d3path):
        webbrowser.open(os.path.abspath(d3path), new=2)
    print('-----------------------Done!-----------------------')

#        df=pd.read_csv('D://stack/TOOLBOX_PY/DATA/OTHER/titanic/titanic_train.csv')
#        df.columns
        
        # out = hnet.main(df, alpha=alpha, dropna=dropna)
        # G = hnet.plot_d3graph(out, savepath=SAVEPATH)
#        headernames=','.join(df.columns.values)
#        headernames=[]
#        for headername in df.columns.values:
#            headernames.append({'label':str(headername),'value':headername})

    return(('%s done!' %(filename)))

#%% Split filepath into dir, filename and extension
def splitpath(filepath, rem_spaces=False):
    [dirpath, filename]=os.path.split(filepath)
    [filename,ext]=os.path.splitext(filename)
    if rem_spaces:
        filename=filename.replace(' ','_')
    return(dirpath, filename, ext)

#%% Saving file
def save_file(name, content, savepath):
    # Decode and store file uploaded with plotly dash
    print('[HNET-GUI] Saving uploaded file..')
    data=content.encode('utf8').split(b";base64,")[1]
    filepath = os.path.join(savepath,name)
    with open(filepath, "wb") as fp:
        fp.write(base64.decodebytes(data))
    return(filepath)

#%% Check input params
def check_input(uploaded_filenames, uploaded_file_contents, y_min, alpha, k, excl_background, perc_min_num, specificity, multtest):
    runtxt=[]
    runOK=True
    dropna=[True]

    try:
        k=np.int(k)
    except:
        runtxt.append('k is a required parameter (k=1)\n')
        #k=1

    try:
        y_min=np.int(y_min)
    except:
        runtxt.append('y_min is a required parameter (y_min=10)\n')
        #y_min=10

    try:
        alpha=np.float(alpha)
    except:
        runtxt.append('alpha is a required parameter (alpha=0.05)\n')
        #alpha=0.05

    try:
        perc_min_num=np.float(perc_min_num)
    except:
        perc_min_num=None

    if specificity==None:
        runtxt.append('specificity is a required parameter (specificity=medium)\n')
        #specificity='medium'

    if len(dropna)>0:
        dropna=True
    else:
        dropna=False

    if excl_background=='':
        excl_background=None
    
    if (uploaded_filenames is None) or (uploaded_file_contents is None):
        runtxt.append('Input file is required\n')

    if len(runtxt)>0:
        runOK=False
    
    out=dict()
    out['k']=k
    out['dropna']=dropna
    out['y_min']=y_min
    out['alpha']=alpha
    out['multtest']=multtest
    out['perc_min_num']=perc_min_num
    out['specificity']=specificity
    out['excl_background']=excl_background
    out['uploaded_filenames']=uploaded_filenames
    out['uploaded_file_contents']=uploaded_file_contents
    return(out, runOK, runtxt)
    
#%% Main
if __name__ == "__main__":
    webbrowser.open('http://127.0.0.1:8050/', new=0, autoraise=True)
    app.run_server(debug=True, use_reloader=True)
    #app.run_server(debug=True, port=8888)