""" This function makes a HEATMAP using the input-dataset, including clustering if desired
	A = imagesc(data, <optional>)

 INPUT:
   data:           Numpy or Pandas array
                   Rows x Cols

 OPTIONAL

   labxcol:        list: List of strings that represents that data columns
                   ['aap','boom','mies']

   labxrow:        list: List of strings that represents that data rows
                   ['aap','boom','mies']

   norm:           Boolean [0,1]: Normalize data per column 
                   [1]: Yes
                   [0]: No (default)

   annot:          Boolean [True,False]: Show per cell the value (does not work with clustering)
                   True: 
                   False: (default)

   cmap:           Boolean [0,1]: Colormap
                   https://matplotlib.org/examples/color/colormaps_reference.html
                   'coolwarm'
                   'bwr'        Blue-white-red (default)
                   'RdBu'       Red-white-Blue
                   'binary' or 'binary_r'
                   'seismic'    Blue-white-red 
                   'rainbow'
                   'Blues'      white-to-blue
                   'Reds'
                   'Pastel1'    Discrete colors
                   'Paired'     Discrete colors
                   'Set1'       Discrete colors

   dpi:            Integer: Resolution figure
                   [100] (default)

   labxTop:        Boolean [0,1]:  Plot labels at top of figure
                   [1]: Yes (default)
                   [0]: No 

   xtickRot:       Integer [0-360]:  Orientation of the labels
                   [45]: (default)

   ytickRot:       Integer [0-360]:  Orientation of the labels
                   [0]: (default)

   caxis:          Integer [x,y]:  range of colors with minimum and maximum value
                   [None,None]:    No range (default)
                   [0,1]:          Range between 0-1

   height:         Integer:  Height of figure
                   [5]: (default)

   width:          Integer:  Width of figure
                   [5]: (default)

   colorbar:       Boolean [0,1]
                   [0]: No 
                   [1]: Yes (default)

   cluster:        Boolean [False,True]: Cluster data hierarchical with euclediance distance and compolete linkage 
                   True: Yes
                   False: No (default)

   distance:       String: Distance measure for the clustering 
                   'euclidean' (default)

   linkage:        String: Linkage type for the clustering 
                   'ward' (default)

   savepath:       String: pathname of the file
                   'c:/temp/heatmap.png'

   plottype=       String: Make bokeh plot:
                   'default' = seaborn
                   'all'

   showprogress    Boolean [0,1]
                   [0]: No (default)
                   [1]: Yes

 OUTPUT
	output

 DESCRIPTION
    http://seaborn.pydata.org/generated/seaborn.clustermap.html
    https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
    https://matplotlib.org/examples/color/colormaps_reference.html

 EXAMPLE
   %reset -f
   import pandas as pd
   import numpy as np
   from VIZ.imagesc import imagesc

   data    = pd.DataFrame(np.random.rand(300,444))
   labxcol = data.columns.values.tolist()
   savepath  = "c://temp//magweg//fig.png"
   [fig,out] = imagesc(data, linewidth=0, cluster=1)

   data    = pd.DataFrame(np.random.rand(3,4))
   labxcol = ['aap', 'boom', 'mies','banaan'] 
   labxrow = ['aap', 'boom', 'mies']
   [A,_] = imagesc(data.values, labxcol=labxcol, labxrow=labxrow, cluster=1)
   [_,_] = imagesc(data.iloc[A['row_reorderd'],A['col_reorderd']], cluster=0)

 SEE ALSO
   hist, donutchart

"""
#--------------------------------------------------------------------------
# Name        : imagesc.py
# Version     : 1.0
# Author      : E.Taskesen
# Date        : Sep. 2017
#--------------------------------------------------------------------------

#%% Libraries
import matplotlib.pyplot as plt
import pandas as pd
#import numpy as np
import seaborn as sns
#from VIZ.savefig import savefig
#import plotly.figure_factory as ff

#%%
def imagesc(data, labxcol='', labxrow='', cluster=False, norm=0, z_score=None, standard_scale=None, cmap='coolwarm', dpi=100, labxTop=1, xtickRot=90, ytickRot=0, width=10, height=10, savepath='', colorbar=1, distance='euclidean', linkage='ward', linewidth=0.1, caxis=[None,None], annot=False, linecolor='000000', plottype='default', showprogress=0):
	#%% DECLARATIONS
    out ={};
    fig ='';
    # Make dictionary to store Parameters
    Param = {}
    Param['showprogress'] = showprogress
    Param['labxcol']      = labxcol
    Param['labxrow']      = labxrow
    Param['cluster']      = cluster
    Param['norm']         = norm
    Param['cmap']         = cmap
    Param['dpi']          = dpi
    Param['labxTop']      = labxTop
    Param['xtickRot']     = xtickRot
    Param['ytickRot']     = ytickRot
    Param['height']       = height
    Param['width']        = width
    Param['colorbar']     = colorbar
    Param['savepath']     = savepath # c:/temp/heatmap.png
    Param['distance']     = distance # correlation
    Param['linkage']      = linkage
    Param['linewidth']    = linewidth
    Param['linecolor']    = linecolor
    Param['caxis']        = caxis
    Param['annot']        = annot
    Param['plottype']     = plottype

    #%% Check cols and rows with linewidth
    if ( (data.shape[0]>100) or (data.shape[1]>100) ) and (Param['linewidth']>0):
        print('>WARNING: Plot will be poorly visible if [linewidth>0] with rows/columns>100. Set linewidth=0 to adjust. [auto-adjusting...]' )
        Param['linewidth']=0
    
    #%% Check data-type
    if (labxcol!=''):
        if (data.shape[1]!=len(labxcol)):
            print('Number of column labels does not match with datamatrix <Fix and RETURN>')
            return(fig, out)
    if (labxrow!=''):
        if (data.shape[0]!=len(labxrow)):
            print('Number of column labels does not match with datamatrix <Fix and RETURN>')
            return(fig, out)

    # Convert pandas dataFrame to numpy-array
    if 'numpy' in str(type(data)):
        data=pd.DataFrame(data)

    #%% Set labels
    # Collect column names from dataframe
    if ('pandas' in str(type(data))) and (labxcol==''):
        labxcol=data.columns.values.tolist()

    # Collect index names from dataframe
    if ('pandas' in str(type(data))) and (labxrow==''):
        labxrow=data.index.tolist()

    if 'numpy' in str(type(labxcol)):
        colOK=1
    elif (type(labxcol)==list):
        colOK=1
    else:
        colOK=0
        # print("WARNING: labxcol must be of type LIST <return>.")
    

    if 'numpy' in str(type(labxrow)):
        rowOK=1
    elif (type(labxrow)==list):
        rowOK=1
    else:
        rowOK=0
        # print("WARNING: labxrow must be of type LIST <return>.")
    
    if colOK==1:
        data.columns = labxcol
    if rowOK==1:
        data.index = labxrow

    #%% Normalize data columns
    if Param['norm']==1:
        data = (data - data.mean()) / (data.max() - data.min())
    
    #%%
#    figure = ff.create_annotated_heatmap(
#        z=data.values,
#        x=list(data.columns),
#        y=list(data.index),
#        annotation_text=data.round(2).values,
#        showscale=True)

    #%% Show data
    if Param['cluster']:
        # With clustering
        sns.set(color_codes=True)

        # Cluster data
        sns.set(font_scale=1.4) # Scale all fonts in the figure with factor 1.4
        g = sns.clustermap(data, method=Param['linkage'], metric=Param['distance'], col_cluster=True, row_cluster=True, linecolor=Param['linecolor'], linewidths=Param['linewidth'], cmap=Param['cmap'], standard_scale=standard_scale, z_score=z_score, figsize=(Param['width'], Param['height']), vmin=Param['caxis'][0], vmax=Param['caxis'][1], annot_kws={"size": 16})
#        g = sns.clustermap(data.corr(), center=0, cmap=Param['cmap'],row_colors=network_colors, col_colors=network_colors,linewidths=.75, figsize=(Param['width'], Param['height']))

        plt.setp(g.ax_heatmap.get_yticklabels(), rotation=Param['ytickRot'])  # For y axis
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=Param['xtickRot']) # For x axis

#        for a in g.ax_row_dendrogram.collections:
#            a.set_linewidth(Param['linewidth'])
#        for a in g.ax_col_dendrogram.collections:
#            a.set_linewidth(Param['linewidth'])
            
        fig = g.fig
        
        # Get order
        col_reorderd = g.dendrogram_col.reordered_ind
        row_reorderd = g.dendrogram_row.reordered_ind

        # re-Order data based on clustering
        data = data.iloc[row_reorderd,col_reorderd]
        out['col_reorderd']      = col_reorderd
        out['row_reorderd']      = row_reorderd
    else:
        # Plot
        # Make Figure
        [fig,ax]=plt.subplots(figsize=(Param['width'],Param['height']))
        
        # Make heatmap
#        if isinstance(Param['cmap'], str):
#            Param['cmap']=sns.color_palette(Param['cmap'])
        ax = sns.heatmap(data, cmap=Param['cmap'], linewidths=Param['linewidth'], vmin=Param['caxis'][0], vmax=Param['caxis'][1], annot=Param['annot'], linecolor=Param['linecolor'])
#        ax = sns.heatmap(data, linewidths=Param['linewidth'], vmin=Param['caxis'][0], vmax=Param['caxis'][1], annot=Param['annot'])
#        sns.palplot(Param['cmap'])
#        sns.set_style("dark")

        # Set labels
        ax.set_xticklabels(labxcol, rotation=Param['xtickRot'], ha='center', minor=False)
        ax.set_yticklabels(labxrow, rotation=Param['ytickRot'], ha='right', minor=False)
               
        # Set appropriate font and dpi
        sns.set(font_scale=1.2)
        sns.set_style({"savefig.dpi": Param['dpi']})
 
       # set the x-axis labels on the top
        if Param['labxTop']==1:
            ax.xaxis.tick_top()
    
        # rotate the x-axis labels
        #plt.xticks(rotation=Param['xtickRot'])
        #plt.yticks(rotation=Param['ytickRot'])
        
        # get figure (usually obtained via "fig,ax=plt.subplots()" with matplotlib)
        fig = ax.get_figure()
    
        # specify dimensions
#        fig.set_size_inches(Param['width'], Param['height'])
        
        #if Param['colorbar']==1:
        #ax.set_xticklabels(row_labels, minor=False)
        #ax.set_yticklabels(column_labels, minor=False)
        #fig.set_aspect('auto')

    # Range figure
#    if Param['caxis']!=[]:
#        plt.clim(Param['caxis'][0],Param['caxis'][1]) 
    
    # Write figure to path
    plt.show()
    #savefig(fig, Param['savepath'])

    # END
    return(out,fig)

