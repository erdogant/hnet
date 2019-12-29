#%% Libraries
# pip install colour dash dash_bootstrap_components dash_core_components dash_html_components
import os
import base64
from pathlib import Path
# popular libraries
import numpy as np
import pandas as pd
#import Core
# Front-end
# import dash_dangerously_set_inner_html
# import codecs
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import webbrowser
# Network center panel
import networkx as nx
import plotly.graph_objs as go
from colour import Color
# from textwrap import dedent as d
# import json
# HNet
import hnet as hnet
import hnet.helpers.picklefast as picklefast

# progressbar
# import tkinter as tk
# from tkinter import ttk
# global labels
global TMP_DIRECTORY, HNET_DIR_STABLE, NETWORK_LAYOUT, HNET_OUT
# global edge1, node1

#%% Initializatoin
TMP_DIRECTORY     = './results/tmp/'
# HNET_DIR_TMP      = './results/tmp/'
HNET_DIR_STABLE   = './results/stable/'
# BACKGROUND_IMAGE  = 'url(./static/background.jpg)'
TMP_DIRECTORY_DEL = False

# Create directories
path=Path(TMP_DIRECTORY)
path.mkdir(parents=True, exist_ok=True)
# path=Path(HNET_DIR_TMP)
# path.mkdir(parents=True, exist_ok=True)
path=Path(HNET_DIR_STABLE)
path.mkdir(parents=True, exist_ok=True)

# At initialization remove content in the tmp directory
if TMP_DIRECTORY_DEL:
    print('[HNET-GUI] Cleaning files from tmp directory..')
    remfiles=os.listdir(TMP_DIRECTORY)
    for remfile in remfiles:
        if os.path.isfile(remfile): os.remove(os.path.join(TMP_DIRECTORY,remfile))

# Extract HNet results from tmp and stable directories
# HNET_PATH_STABLE=get_hnetpath(HNET_DIR_STABLE)
HNET_PATH_STABLE=[]
for i in os.listdir(HNET_DIR_STABLE):
    getdir=os.path.join(HNET_DIR_STABLE,i)
    if os.path.isdir(getdir):
        HNET_PATH_STABLE.append({'label':i,'value':getdir})


# HNET_PATH_TMP    = [{'label':i,'value':os.path.join(HNET_DIR_TMP,i)} for i in os.listdir(HNET_DIR_TMP)]
# HNET_PATH_TOTAL  = HNET_PATH_STABLE + HNET_PATH_TMP

#%% NETWORK PROPERTIES
# ALPHA_SCORE=[0,1000]
ALPHA_SCORE=0
NODE_NAME=''
NETWORK_LAYOUT='fruchterman_reingold'
HNET_OUT=None
edge1=None
node1=None

#%%
# df=pd.read_csv('D://stack/TOOLBOX_PY/DATA/OTHER/titanic/titanic_train.csv')
# HNET_OUT=hnet.main(df)
#labels=[{'label':i,'value':i} for i in df.columns.unique()]
#if 'labels' not in globals():
#    labels=[{'label':'','value':''}]
#    print('setup labels first time')

#%%
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#app.css.append_css({'external_url':external_stylesheets}) # Required for making columns
app.scripts.config.serve_locally = True
app.title = "HNet: Graphical Hypergeometric Networks by inference"

# app.scripts.append_script({"external_url": ['https://d3js.org/d3.v3.min.js']})

# Normally Dahs creates its own Flas server internally but by creating our own, we cancreate a route for downloading files
# server=Flask(__name__)
#app=dash.Dash(server=server)
#app.css.append_css({'external_url':'./static/bWLwgP.css'}) # Required for making columns

#%% Start d3 network in browser
def boot_d3network_in_browser(dropdown_path, who=''):
    print('D3PATH: WHO: [%s]' %who)
    d3path=''
    if dropdown_path!=None:
        if os.path.isfile(get_d3path(dropdown_path)):
            print(d3path)
            d3path=os.path.abspath(get_d3path(dropdown_path))
            # open in webbrowser
            webbrowser.open(d3path, new=2)
            # https://plot.ly/python/network-graphs/
    return(d3path)

#%% Split filepath into dir, filename and extension
# def splitpath(filepath, rem_spaces=False):
#     [dirpath, filename]=os.path.split(filepath)
#     [filename,ext]=os.path.splitext(filename)
#     if rem_spaces:
#         filename=filename.replace(' ','_')
#     return(dirpath, filename, ext)

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

#%% List result directories
def get_hnetpath(HNET_DIR_STABLE):
    HNET_PATH_STABLE=[]
    for i in os.listdir(HNET_DIR_STABLE):
        getdir=os.path.join(HNET_DIR_STABLE,i)
        if os.path.isdir(getdir):
            HNET_PATH_STABLE.append({'label':i,'value':getdir})
    return(HNET_PATH_STABLE)

# Make path for d3js
def get_d3path(filepath):
    return(os.path.join(filepath,'index.html'))

def get_pklpath(filepath):
    return(os.path.join(filepath,'hnet.pkl'))

#%% Create Network
# def network_graph_hnet(alphaRange, NodeToSearch, network_layout, HNET_OUT=None):
#     https://towardsdatascience.com/python-interactive-network-visualization-using-networkx-plotly-and-dash-e44749161ed7
#     import plotly.graph_objects as go

#     if isinstance(HNET_OUT, type(None)):
#         edge_trace = go.Scatter(x=[0], y=[0], line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
#         node_trace = go.Scatter(x=[0], y=[0])
#     else:
#         G = HNET_OUT['G']['G']
#         # G = nx.random_geometric_graph(200, 0.125)
        
#         # Create edge traces
#         edge_x = []
#         edge_y = []
#         for edge in G.edges():
#             x0, y0 = HNET_OUT['G']['pos'][edge[0]]
#             x1, y1 = HNET_OUT['G']['pos'][edge[0]]
#             # x0, y0 = G.nodes[edge[0]]['pos']
#             # x1, y1 = G.nodes[edge[1]]['pos']
#             edge_x.append(x0)
#             edge_x.append(x1)
#             # edge_x.append(None)
#             edge_y.append(y0)
#             edge_y.append(y1)
#             # edge_y.append(None)
        
#         edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
        
#         node_x = []
#         node_y = []
#         for node in G.nodes():
#             # x, y = G.nodes[node]['pos']
#             x, y = HNET_OUT['G']['pos'][node]
#             node_x.append(x)
#             node_y.append(y)
        
#         # Create node traces
#         node_trace = go.Scatter(
#             x=node_x, y=node_y,
#             mode='markers',
#             hoverinfo='text',
#             marker=dict(
#                 showscale=True,
#                 # colorscale options
#                 #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
#                 #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
#                 #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
#                 colorscale='YlGnBu',
#                 reversescale=True,
#                 color=[],
#                 size=10,
#                 colorbar=dict(
#                     thickness=15,
#                     title='Node Connections',
#                     xanchor='left',
#                     titleside='right'
#                 ),
#                 line_width=2))
    
#         # Create node traces
#         node_adjacencies = []
#         node_text = []
#         for node, adjacencies in enumerate(G.adjacency()):
#             node_adjacencies.append(len(adjacencies[1]))
#             node_text.append('# of connections: '+str(len(adjacencies[1])))
        
#         node_trace.marker.color = node_adjacencies
#         node_trace.text = node_text
        
#     # Combine edge and node traces in figure
#     figure = go.Figure(data=[edge_trace, node_trace],
#                  layout=go.Layout(
#                     title='<br>Network graph made with Python',
#                     titlefont_size=16,
#                     showlegend=False,
#                     hovermode='closest',
#                     margin=dict(b=20,l=5,r=5,t=40),
#                     annotations=[ dict(
#                         text="HNET graph",
#                         showarrow=True,
#                         xref="paper", yref="paper",
#                         x=0.005, y=-0.002 ) ],
#                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
#                     )
#     # figure.show()
#     return(figure)

#%% Create Network
def network_graph(alphaRange, NodeToSearch, network_layout, HNET_OUT=None):
    # if (isinstance(edge1, type(None))) and (isinstance(node1, type(None))):
    #     print('demo case')
    edge1 = pd.read_csv(os.path.join(TMP_DIRECTORY+'edge1.csv'))
    node1 = pd.read_csv(os.path.join(TMP_DIRECTORY+'node1.csv'))

    # filter the record by alpha score, to enable interactive control through the input box
    #print(alphaRange)
    I = edge1['Weight'].values >= np.min(alphaRange)
    # I = edge1['Weight'].values >= alphaRange
    edge1 = edge1.loc[I,:]
    edge1.reset_index(drop=True, inplace=True)

    # I = np.isin(node1['NodeName'].values,  np.unique(list(edge1['Source'])+list(edge1['Target'])))
    # node1 = node1.loc[I,:]
    nodeSet= np.unique(list(edge1['Source'])+list(edge1['Target']))
    # nodeSet=set() # contain unique Nodes
    # for index in edge1.index:
    #     nodeSet.add(edge1['Source'][index])
    #     nodeSet.add(edge1['Target'][index])

    # to define the centric point of the networkx layout
    shells=[]
    shell1=[]
    shell1.append(NodeToSearch)
    shells.append(shell1)
    shell2=[]
    for node in nodeSet:
        if node!=NodeToSearch:
            shell2.append(node)
    shells.append(shell2)
    
    # import network as network
    # edge1 = edge1[['Source','Target','Weight']]
    # G = network.df2G(node1.set_index('NodeName'), edge1)
    G = nx.from_pandas_edgelist(edge1, 'Source', 'Target', ['Source', 'Target', 'Weight'], create_using=nx.MultiDiGraph())
    
    # # hnet.plot_network
    # HNET_OUT=get_pklpath(hnet_path)
    # if hnet_path!='hnet.pkl':
        # print('LOAD HNet ')
        # hnet_out = picklefast.load(hnet_path)
        # G = hnet_out['G']['G']
    
    # os.path.isfile(hnet_path)
    # if isinstance(hnet_path, type(None)) or (not os.path.isfile(hnet_out)):
    #     G = nx.from_pandas_edgelist(edge1, 'Source', 'Target', ['Source', 'Target', 'Weight'], create_using=nx.MultiDiGraph())
    # else:
    #     print('Get HNet Graph: %s' %(hnet_path))
    #     G = hnet_out['G']['G']
    #     pos = hnet_out['G']['pos']

    nx.set_node_attributes(G, node1.set_index('NodeName')['Label'].to_dict(), 'Label')
    nx.set_node_attributes(G, node1.set_index('NodeName')['Type'].to_dict(), 'Type')

    # pos = nx.layout.spring_layout(G)
    # pos = nx.layout.circular_layout(G)
    # nx.layout.shell_layout only works for more than 3 nodes
    # print(len(NodeToSearch))

    if len(shell2)>1 and isinstance(NodeToSearch, type(None)) and NodeToSearch=='':
    # if len(shell2)>1:
        pos = nx.layout.shell_layout(G, shells, scale=0.4, iterations=50)
    else:
        if network_layout=='fruchterman_reingold':
            pos = nx.layout.fruchterman_reingold_layout(G, scale=0.4, iterations=50)
        elif network_layout=='kamada_kawai':
            pos = nx.layout.kamada_kawai_layout(G, scale=0.4)
        elif network_layout=='circular':
            pos = nx.layout.circular_layout(G, scale=0.4)
        elif network_layout=='planar':
            pos = nx.layout.planar_layout(G, scale=0.4)
        else:
            pos = nx.layout.spring_layout(G, scale=0.4, iterations=50)

    for node in G.nodes:
        G.nodes[node]['pos'] = list(pos[node])


    # if not isinstance(HNET_OUT, type(None)):
    #     # print('Get HNET_PATH Graph: %s' %(HNET_OUT))
    #     print('Get HNET_PATH Graph!!!')
    #     del G
    #     print(HNET_OUT['G']['pos'])
    #     G = HNET_OUT['G']['G']
    #     pos = HNET_OUT['G']['pos']

    # IF EMPTY
    if len(shell2)==0:
        print('Empty number of nodes selected.')
        traceRecode = []  # contains edge_trace, node_trace, middle_node_trace

        node_trace = go.Scatter(x=tuple([1]), y=tuple([1]), text=tuple([str(NodeToSearch)]), textposition="bottom center",
                                mode='markers+text',
                                marker={'size': 50, 'color': 'LightSkyBlue'})
        traceRecode.append(node_trace)

        node_trace1 = go.Scatter(x=tuple([1]), y=tuple([1]),
                                mode='markers',
                                marker={'size': 50, 'color': 'LightSkyBlue'},
                                opacity=0)
        traceRecode.append(node_trace1)

        figure = {
            "data": traceRecode,
            "layout": go.Layout(title='Network', showlegend=False,
                                margin={'b': 40, 'l': 40, 'r': 40, 't': 40},
                                xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                                yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                                height=600
                                )}
        return(figure)

    else:
        traceRecode = []  # contains edge_trace, node_trace, middle_node_trace
        ############################################################################################################################################################
        colors = list(Color('lightcoral').range_to(Color('darkred'), len(G.edges())))
        colors = ['rgb' + str(x.rgb) for x in colors]
    
        for index, edge in enumerate(G.edges):
            x0, y0 = G.nodes[edge[0]]['pos']
            x1, y1 = G.nodes[edge[1]]['pos']
            weight = float(G.edges[edge]['Weight']) / max(edge1['Weight']) * 10
            trace = go.Scatter(x=tuple([x0, x1, None]), y=tuple([y0, y1, None]),
                               mode='lines',
                               line={'width': weight},
                               marker=dict(color=colors[index]),
                               line_shape='spline',
                               opacity=1)
            traceRecode.append(trace)

        ###############################################################################################################################################################
        node_trace = go.Scatter(x=[], y=[], hovertext=[], text=[], mode='markers+text', textposition="bottom center", hoverinfo="text", marker={'size': 50, 'color': 'LightSkyBlue'})
    
        for node in G.nodes():
            x, y = G.nodes[node]['pos']
            hovertext = "Label: " + str(G.nodes[node]['Label']) + "<br>" + "Label: " + str(G.nodes[node]['Type'])
            text = node#node1['NodeName'][index]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            node_trace['hovertext'] += tuple([hovertext])
            node_trace['text'] += tuple([text])
    
        traceRecode.append(node_trace)
        ################################################################################################################################################################
        middle_hover_trace = go.Scatter(x=[], y=[], hovertext=[], mode='markers', hoverinfo="text", marker={'size': 20, 'color': 'LightSkyBlue'}, opacity=0)
    
        for edge in G.edges:
            x0, y0 = G.nodes[edge[0]]['pos']
            x1, y1 = G.nodes[edge[1]]['pos']
            hovertext = "From: " + str(G.edges[edge]['Source']) + "<br>" + "To: " + str(
                G.edges[edge]['Target']) + "<br>" + str(
                G.edges[edge]['Weight']) + "<br>"
            middle_hover_trace['x'] += tuple([(x0 + x1) / 2])
            middle_hover_trace['y'] += tuple([(y0 + y1) / 2])
            middle_hover_trace['hovertext'] += tuple([hovertext])
    
        traceRecode.append(middle_hover_trace)
        #################################################################################################################################################################
        figure = {
            "data": traceRecode,
            "layout": go.Layout(title='Selected network', showlegend=False, hovermode='closest',
                                margin={'b': 40, 'l': 40, 'r': 40, 't': 40},
                                xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                                yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                                height=600,
                                clickmode='event+select',
                                annotations=[
                                    dict(
                                        ax=(G.nodes[edge[0]]['pos'][0] + G.nodes[edge[1]]['pos'][0]) / 2,
                                        ay=(G.nodes[edge[0]]['pos'][1] + G.nodes[edge[1]]['pos'][1]) / 2, axref='x', ayref='y',
                                        x=(G.nodes[edge[1]]['pos'][0] * 3 + G.nodes[edge[0]]['pos'][0]) / 4,
                                        y=(G.nodes[edge[1]]['pos'][1] * 3 + G.nodes[edge[0]]['pos'][1]) / 4, xref='x', yref='y',
                                        showarrow=True,
                                        arrowhead=3,
                                        arrowsize=4,
                                        arrowwidth=1,
                                        opacity=1
                                    ) for edge in G.edges]
                                )}
        return(figure)
    
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
        # Row 1
        html.Div([html.H5("HNets: Graphical Hypergeometric Network inference")], className="row", style={'textAlign':'left','width':'100%','backgroundColor':'#e0e0e0'}),

        # Row 2 - Column 1
        html.Div([

            html.Div([
                html.H6('HNet input parameters'),
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
                html.Div(id="message-box-output"),
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

                # dcc.Dropdown(id='results-id', options=[{'label':i,'value':os.path.join(HNET_DIR_STABLE,i)} for i in os.listdir(HNET_DIR_STABLE)], value='', style={"width": "100%"}),
                dcc.Dropdown(id='opt-dropdown', options=get_hnetpath(HNET_DIR_STABLE), value='', style={"width": "100%"}),
                # dcc.Dropdown(id='opt-dropdown', value='', style={"width": "100%"}),
                # html.Div(id="results-output")

            ], className="three columns", style={"margin":"0px", "width": "15%", "border":"1px black solid", "height": "700px",'backgroundColor':''}),

            # ROWS 2 - COLUMN 2 --------------------- CENTER PANEL: NETWORK ------------------ 
            html.Div(className="three columns", children=[dcc.Graph(id="hnet-graph", figure=network_graph(ALPHA_SCORE, NODE_NAME, NETWORK_LAYOUT))], 
                     style={"margin":"0px","width": "65%", "height": "700px","border":"1px black solid"} ),

            # ROW 2 - COLUMN 3 -------------------- RIGHT PANEL: CONTROLS -------------------
            html.Div([html.H6('Network Settings'),

                dcc.Dropdown(id='network-layout-id',
                    options=[
                        {'label': 'kamada_kawai', 'value': 'kamada_kawai'},
                        {'label': 'planar', 'value': 'planar'},
                        {'label': 'circular', 'value': 'circular'},
                        {'label': 'shell', 'value': 'shell'},
                        {'label': 'spring', 'value': 'spring'},
                        {'label': 'fruchterman_reingold', 'value': 'fruchterman_reingold'},
                        {'label': 'graph', 'value': 'graph'}
                    ],
                    value='fruchterman_reingold', style={"width": "100%"}),

                dcc.Input(id='alpha-slider-id', placeholder='slider', type='text', value=0, style={"width": "100%"}), 
                html.Div(id="output-container-range-slider"),
                dcc.Input(id='node-id', placeholder='Node Name', type='text', style={"width": "100%"}),
                # html.Div(dcc.Input(id='message-box', type='text', style={"width": "100%"})),
                html.Button('Make d3graph', id='button', style={"width": "100%"}),
                html.Div(id='output-container-button',children='Enter a value and press submit', style={"width": "100%"}),
                # html.Button('Make d3graph!', id='button-id'),                
                # html.Div(id='output-button', children='Enter a value and press submit'),
                # html.Div(id="output"),
                ], className="three columns", style={"margin":"0px","width": "15%", "height": "700px","border":"1px black solid"}),
            

        ], className="row", style={"width": "100%"}),

        # html.Div(
        #     [dcc.Interval(id="progress-interval", n_intervals=0, interval=500),
        #      dbc.Progress(id="progress")],
        #     className="row", style={"width": "95%","border":"1px black solid"}
        #     ),


    ], className="row", style={"width": "100%"} #style={"max-width": "500px"},
)

#%% Run webpage
app.layout = html.Div([GUIelements])

#%%
# @app.callback(
#     [Output("progress", "value"), Output("progress", "children")],
#     [Input("progress-interval", "n_intervals")],
# )
# def update_progress(n):
#     https://dash-bootstrap-components.opensource.faculty.ai/l/components/progress
#     # check progress of some background process, in this example we'll just
#     # use n_intervals constrained to be in 0-100
#     progress = min(n % 110, 100)
#     # only add text after 5% progress to ensure text isn't squashed too much
#     return progress, f"{progress} %" if progress >= 5 else ""

#%% Callback for button (right panel)
@app.callback(
    Output('output-container-button', 'children'),
    [Input('button', 'n_clicks'), Input("opt-dropdown","value")])
def button_click_action(n_clicks, dropdown_path):
    if dropdown_path==None: dropdown_path=''
    d3path=boot_d3network_in_browser(dropdown_path, who='button')
    if d3path=='': d3path='Select first using the dropdown-box!'
    print(d3path)

#%% Callback for plotly (right-side) components
@app.callback(
    Output('hnet-graph', 'figure'),
    [Input('alpha-slider-id', 'value'), Input('node-id', 'value'), Input("opt-dropdown","value"), Input('network-layout-id', 'value')])
def update_output(alpha_limit, node_name, dropdown_path, network_layout):
    # Change input variables
    if alpha_limit=='': alpha_limit=0
    if alpha_limit==None: alpha_limit=0
    if network_layout=='': network_layout='fruchterman_reingold'
    if network_layout==None: network_layout='fruchterman_reingold'
    if dropdown_path==None: dropdown_path=''
    alpha_limit=np.int(alpha_limit)
    ALPHA_SCORE = np.int(alpha_limit)
    NODE_NAME = node_name
    NETWORK_LAYOUT = network_layout
    HNET_OUT = None
    # edge1=None
    # node1=None

    # Print selection dropdownbox
    # print(get_d3path(dropdown_path))
    # print(get_pklpath(dropdown_path))
    print('Dropdown_path path: %s\n' %(dropdown_path))
    print('Node name selected : %s\n' %(node_name))
    print('Node layout selected : %s\n' %(network_layout))
    # Load data
    if os.path.isfile(get_pklpath(dropdown_path)):
        print('Load picke file')
        HNET_OUT=picklefast.load(get_pklpath(dropdown_path))
        # HNET_OUT=hnet_out
        #print(df['simmatLogP'])

        df_edges=HNET_OUT['simmatLogP'].stack().reset_index()
        df_edges.columns=['Source', 'Target', 'Weight']
        df_edges['Weight']=df_edges['Weight'].astype(float)
        edge1 = df_edges.loc[df_edges['Weight']>0,:]
        node1 = pd.DataFrame(np.unique(df_edges[['Source','Target']].values.reshape(-1)), columns=['NodeName'])
        node1['Label']=node1['NodeName']
        node1['Type']=''
        
        # Write to disk
        edge1.to_csv(os.path.join(TMP_DIRECTORY+'edge1.csv'), index=False)
        node1.to_csv(os.path.join(TMP_DIRECTORY+'node1.csv'), index=False)
        # ALPHA_SCORE=[0,np.max(edge1['Weight'].values)]
        
    
    # Data read
    # edge1 = pd.read_csv(os.path.join(TMP_DIRECTORY+'edge1.csv'))
    # node1 = pd.read_csv(os.path.join(TMP_DIRECTORY+'node1.csv'))
    
    # Make graph
    # if (not isinstance(edge1, type(None))) and (not isinstance(node1, type(None))):
    hnet_graph=network_graph(alpha_limit, node_name, network_layout, HNET_OUT=HNET_OUT)
    # else:
    #     print('Select one first in dropdown!')
    # Return to screen
    return(hnet_graph)
    # to update the global variable of ALPHA_SCORE and NODE_NAME

#%% Callback for center screen
# @app.callback(
#     Output("results-output", "children"),
#     [Input("opt-dropdown","value")])
# def load_results(dropdown_path):
#     """Save uploaded files and regenerate the file list."""
#     # print(dropdown_path)
#     # d3path=boot_d3network_in_browser(dropdown_path)
#     # Return
#     return(dropdown_path)

#%% Callback for HNet menu (left-side)
@app.callback(
    [Output("message-box-output", "children")],
    # Output('opt-dropdown', 'options'),
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
    # Check input parameters
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
    [_,filename, _] = hnet.path_split(args['uploaded_filenames'])
    savepath        = os.path.join(HNET_DIR_STABLE,filename+'_'+str(args['y_min'])+'_'+str(args['k'])+'_'+str(args['multtest'])+'_'+str(args['specificity'])+'_'+str(args['perc_min_num'])+'_'+str(args['excl_background'])+'/')
    d3path          = get_d3path(savepath)
    pklpath         = get_pklpath(savepath)

    print('filepath %s' %(filepath))
    print('savepath %s' %(savepath))
    print('d3path %s' %(d3path))
    print('pklpath %s' %(pklpath))

    # Progressbar
    # mGUI=tk.Tk()
    # mGUI.title('HNET '+ filename)
    # progressbar = ttk.Progressbar(mGUI, orient='horizontal', length=300)
    # progressbar.pack(side=tk.TOP)
    # progressbar.config(mode='determinate')
    # progressbar.update()
    # progressbar['value']=0
    # progressbar['maximum']=5
    # # Set starting position
    # progressbar['value']=progressbar['value']+1 # update bar
    # progressbar.update() # update gui

    # Make directory
    if not os.path.isdir(savepath):
        os.mkdir(savepath)
    # Make D3js path
    if not os.path.isfile(d3path):
        # Read file
        df = pd.read_csv(filepath)
        # Run HNet
        # progressbar['value']=progressbar['value']+1 # update bar
        # progressbar.update() # update gui
        HNET_OUT = hnet.fit(df, alpha=args['alpha'], y_min=args['y_min'], k=args['k'], multtest=args['multtest'], dtypes='pandas', specificity=args['specificity'], perc_min_num=args['perc_min_num'], dropna=args['dropna'], excl_background=args['excl_background'], verbose=3)
        # Save pickle file
        print('SAVING NETWORK FIGURE: %s' %(savepath))
        # progressbar['value']=progressbar['value']+1 # update bar
        # progressbar.update() # update gui
        HNET_OUT['G'] = hnet.plot_network(HNET_OUT, dist_between_nodes=0.4, scale=2, dpi=250, figsize=[30,20], showfig=False, savepath=os.path.join(savepath,'hnet_network.png'))
        # Store pickle file
        print('STORE PICKLE')
        # progressbar['value']=progressbar['value']+1 # update bar
        # progressbar.update() # update gui
        picklefast.save(pklpath, HNET_OUT)
        print('MAKE D3GRAPH')
        _ = hnet.plot_d3graph(HNET_OUT, savepath=savepath, directed=False, showfig=False)
    else:
        print('dir exists, load stuff')
        HNET_OUT=picklefast.load(pklpath)
        
    # Open in browser
    if os.path.isfile(d3path):
        print('OPEN BROWSER')
        webbrowser.open(os.path.abspath(d3path), new=2)
        # boot_d3network_in_browser(d3path, who='upload')

    print('HNET_DIR_STABLE: %s!' %(HNET_DIR_STABLE))
    print('%s done!' %(filename))
    print('-----------------------Done!-----------------------')
    # progressbar['value']=progressbar['maximum'] # update bar
    # progressbar.update() # update gui
    # try:
    #     mGUI.destroy()
    # except:
    #     pass
    return(filename)


#%% Main
if __name__ == "__main__":
    # webbrowser.open('http://127.0.0.1:8050/', new=0, autoraise=True)
    app.run_server(debug=True, use_reloader=True, port=8050)
