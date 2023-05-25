#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function

import pandas as pd
from igraph import *
from ipywidgets import interact

patient = "H030"
# H024
data = pd.read_csv("/home/aimaaral/dev/clonevol/examples/" + patient + ".csv", sep=",")
data = data.drop(data.columns[0], axis=1).dropna(axis='rows')


# H021 has no logic, double check!
# OC005 check ones that has multiple subgraphs


def build_graph(patientdf: pd.DataFrame):
    graph = Graph(directed=True)
    # colors = ('#808080','#7f0000','#006400','#808000','#483d8b','#3cb371','#008b8b','#cd853f','#4682b4','#00008b','#32cd32','#7f007f','#b03060','#ff4500','#ff8c00','#00ff00','#00fa9a','#8a2be2','#dc143c','#00ffff','#0000ff','#f08080','#ff00ff','#1e90ff','#eee8aa','#ffff54','#dda0dd','#b0e0e6','#ff1493','#7b68ee')
    gcolors = (
        '#b01f35', '#b0bf1a', '#663380', '#ff7008', '#00c20c', '#05f6d5', '#efe9e4', '#696969', '#2b80ff', '#eeee44',
        '#338cc7',
        '#806680', '#803380', '#cd853f', '#4682b4', '#00008b', '#32cd32', '#7f007f', '#b03060')

    dg = patientdf.sort_values(['parent']).reset_index()

    print(patientdf.sort_values(['sample', 'dfs.order']))
    dp = dg.groupby("cluster")
    for group_name, group in dp:
        # print(group)

        # frac = dg.loc[i]['frac']
        # Adding the vertices
        for index, row in group.iterrows():
            parent = int(row['parent'])
            if row['parent'] == -1:
                parent = 0

            hasp = False
            hasc = False
            try:
                graph.vs.find(cluster=parent)
                hasp = True
            except Exception as e:
                pass

            try:
                graph.vs.find(cluster=row['cluster'])
                hasc = True
            except:
                pass

            if hasp == False:
                # Adding the vertex properties
                c = graph.add_vertex()
                p = dg.loc[dg['cluster'] == int(row['parent'])]
                color = "#cccccc"
                if parent != 0:
                    color = p['color'].values[0]

                c["label"] = row['parent']
                c["cluster"] = parent
                c["sample"] = row['sample']
                c["frac"] = group['frac'].max()  # we store maximum fraction to filter clusters later with a threshold
                c['parent'] = parent
                c["color"] = color

            if hasc == False:
                # Adding the vertex properties
                c = graph.add_vertex()
                c["label"] = row['cluster']
                c["cluster"] = row['cluster']
                c["sample"] = row['sample']
                c["frac"] = group['frac'].max()
                c['parent'] = parent
                c["color"] = row['color']

            try:
                i1 = graph.vs.find(cluster=parent)
                i2 = graph.vs.find(cluster=row['cluster'])
                if (i1.index, i2.index) not in graph.get_edgelist()[0:]:
                    graph.add_edge(i1, i2)
                # print(parent,row['cluster'])
            except Exception as e:
                print("Vertex not found")

            # To display the Igraph
            # layout = g.layout("tree")

    graph.delete_vertices(0)
    # print(graph.get_all_simple_paths(graph.vs.find(cluster=1),graph.vs.find(cluster=4),mode='all'))
    plot(graph, bbox=(600, 600), margin=20,layout="tree")
    return graph


# TODO: check Sankey plot to make grouped tentacles https://www.python-graph-gallery.com/sankey-diagram-with-python-and-plotly


# In[4]:


# With transformations and relative values
# TODO: implement root creation in single reusable method. 
# If same site is sampled multiple times during the treatment, connect clones between them, eg. iPer1, rPer1 or r1Asc, r2Asc. 
# (use the method extractPointByClusterColor ti get the bezier starting points)
# If cluster is not end node but included only in interval or relapsed, exclude from root
# If cluster is end node but in multiple samples in same treatment phase, move to root jelly

import drawSvg as draw

from PIL import Image


def extractPointByClusterColor(sx, ex, sy, ey, color, img):
    img = Image.open(img)
    pixels = img.load()
    width, height = img.size

    crangey = {}

    for x in range(sx, ex):
        found = False
        firsty = 0
        lasty = 0
        for y in range(sy, ey):  # this row

            # and this row was exchanged
            # r, g, b = pixels[x, y]

            # in case your image has an alpha channel
            r, g, b, a = pixels[x, y]
            cc = f"#{r:02x}{g:02x}{b:02x}"

            if color == cc and found == False:
                found = True
                firsty = y
            else:
                if color != cc and found == True:
                    lasty = y
                    found = False
                    break

    return firsty, lasty


def addAxes(el):
    for t in range(1, 10):
        l = draw.Line(0, t * 50, 5, t * 50, stroke="black", stroke_width='2')
        el.append(l)
        te = {
            'text': str(t * 50),
            'fontSize': '10',
            'fill': 'black',
            'x': 10,
            'y': t * 50
        }
        # rg.append(draw.Use('rc', 100,100))
        el.append(draw.Text(**te))

    for f in range(1, 20):
        el.append(draw.Line(f * 50, 0, f * 50, 5, stroke="black", stroke_width='2'))
        ty = {
            'text': str(f * 50),
            'fontSize': '10',
            'fill': 'black',
            'x': f * 50,
            'y': 10
        }
        # rg.append(draw.Use('rc', 100,100))
        el.append(draw.Text(**ty))


def removeRootCluster(rootgroup: draw.Group):
    for el in rootgroup.allChildren():
        if isinstance(el, draw.elements.Path) == True:
            if str(el.id).startswith('rgi'):
                print(el)


def moveSampleBox(samplegroup: draw.Group, moveX, moveY):
    samplegroup.args['transform'] = 'translate(' + str(moveX) + ',' + str(moveY) + ')'
    for el in samplegroup.allChildren():
        if isinstance(el, draw.elements.Path) == True:
            if str(el.id).startswith('tnt'):
                args = el.args['d'].split(' ')
                M = args[0].split(',')
                C = args[1].split(',')
                print(el.args['d'])
                Mx = float(M[0][1:]) - moveX
                My = float(M[1]) - moveY

                # TODO: calculate all new bezier points from new x and y OR maybe better use this function to create boxes and tentacles at first place
                Cex = float(C[4])
                Cey = float(C[5])

                if My > 0:
                    bz1sty = My + 50
                else:
                    bz1sty = My - 50

                if My > 0:
                    bz2ndy = Cey + 50
                else:
                    bz2ndy = Cey - 50

                bz1stx = (Mx + moveX)
                bz2ndx = (Cex + moveX)

                newArgs = "M" + str(Mx) + "," + str(My) + " " + str(bz1stx) + "," + str(bz1sty) + "," + str(
                    bz2ndx) + "," + str(bz2ndy) + "," + str(Cex) + "," + str(Cey)
                el.args['d'] = newArgs
                print(el.args['d'])

import pandas as pd
import os
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt


def calcSampleClonalFreqs(models: pd.DataFrame, files):
    intres = []
    augarr = []
    for file in files:
        print(f"processing '{file}'..")
        basename = os.path.basename(file)
        patient = re.sub(f'^([^_]+\\d+)(_v2)?_vaf_(.*)_cellular_freqs\\.csv$', '\\1', basename)

        freqs = pd.read_csv(file, sep = '\t')
        freqs = freqs.loc[freqs['model.num'] == models.loc[models['patient'].str.contains(patient), 'model'].values[0], :]

        unique_samples = freqs['sample.id'].unique()
        unique_clones = np.arange(1, freqs['cloneID'].max()+1)
        values = np.zeros((len(unique_clones), len(unique_samples)))
        rownames = unique_clones.astype(str)
        colnames = unique_samples

        for i, clone in enumerate(unique_clones):
            for j, sample in enumerate(unique_samples):
                clone_sample_freqs = freqs.loc[(freqs['cloneID'] == clone) & (freqs['sample.id'] == sample), 'cell.freq']
                if not clone_sample_freqs.empty:
                    values[i, j] = clone_sample_freqs.iloc[0] / 100.
        assert np.all((0 <= values) & (values <= 1.))
        assert np.all((1-.05 <= np.sum(values, axis = 0)) & (np.sum(values, axis = 0) <= 1+.05))

        # Calculate Kullback-Leibler divergence of the sample’s clonal frequency distribution from the average distribution over all samples of a patient
        # i.e. inter-tumor heterogeneity
        p = (values / np.sum(values, axis = 0)).T
        z = p*np.log(p)
        # Nan to 0
        z[~(p > 0.)] = 0.
        hc = -np.sum(z, axis = 0)
        #Clonal complexity (latter sum)
        c = np.exp(hc)

        #first sum
        q = np.mean(p, axis = 1)
        z = p*np.log(q.reshape(-1, 1))
        z[~(p > 0.)] = 0.
        hu = -np.sum(z, axis = 1)
        u = np.exp(hu)
        # sum over rows (clones)
        n = np.sum(p > 0., axis = 1)

        #Add w in column names
        #p is missing samples as column names because numpy can not handle it, so add cnames to dataframe index
        aug = pd.DataFrame(p, columns=[f'w{i}' for i in range(1, p.shape[1]+1)], index = unique_samples)
        aug.index.names=['sample']
        augarr.append(aug)
        #auf = pd.merge(auf, aug, how="outer", left_index=True, right_index=True).fillna(0)
        #print(auf.dtypes)

        #print(auf)
        # save averaged clonal frequency distributions per sample

        # TODO: add equation coefficients (hc etc) columns and join dataframes (wi means ith clone and value is 𝑝𝑖𝑗 is the normalized frequency)
        #intres.append(pd.DataFrame({'hc': hc, 'c': c, 'hu': hu, 'u': u, 'n': n}, index = unique_samples, columns = aug.columns))
        #print(results)
    auf = pd.concat(augarr, axis=0, ignore_index=False).fillna(0)
    #results = pd.concat(intres, axis=0, ignore_index=False)
    #print(results)

    #auf.to_csv("/home/aimaaral/dev/tumor-evolution-2023/heterogeneity/avg_cfd.csv",sep = '\t')
    #results = reduce(lambda df1, df2: pd.concat([df1, df2], axis = 1, ignore_index = True), results)
    # function toporbind
    return auf



def calcCorrMatrix(d: pd.DataFrame):
    sns.set_theme(style="white")

    # Generate a large random dataset
    #d = pd.read_csv('/home/aimaaral/dev/tumor-evolution-2023/heterogeneity/cellular_freqs.tsv',sep = '\t').set_index("sample").drop(columns=["hc","hu","c","u","n"])

    # TODO: group by patient and generate corrmatrix for each patient separately
    # Compute the correlation matrix
    corr = d.T.corr()
    #print(d.T)

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    hmap = sns.heatmap(corr, cmap=cmap, center=0,
                       square=True, linewidths=.5, cbar_kws={"shrink": .5})
    hmap
    return corr

def composeSimpleJellyBell(graph: Graph, startnode, endnode, height, width, x, y):
    dropouts = set()
    endvertices = set()

    print(graph.get_all_simple_paths(startnode, endnode))

    #endcluster = graph.vs.find(i)['cluster']
    #dropouts.add(endcluster)
    allpaths = graph.get_all_simple_paths(startnode, endnode)

    # print(allpaths)

    k = 1
    root = graph.vs.find(0)
    frac = root['frac']
    cluster = root['cluster']
    # moves 2nd control points of Bezier curves downwards
    # root = draw.Arc(cx=rx, cy=ry, r=rootrad, startDeg=90, endDeg=270, fill=colors[cluster.astype(int)])

    elementids = []
    clones: draw.Path = []
    clist = []

    rg = draw.Group(id='root2')

    # offy=(k-1)*20
    h = height
    csx = x
    csy = y
    cex = x + width
    cey = y+h
    cc1x = csx + cex / 3
    cc1y = cey / 10
    cc2x = cex - cex / 5
    cc2y = h + 20

    rcolor = data.loc[data['cluster'] == cluster]['color'].values[0]
    rpu = draw.Path(fill=rcolor, fill_opacity=100.0)
    rpu.M(csx, csy)  # Start path at point
    rpu.C(cc1x, csy + cc1y, cc2x, csy+cc2y, cex, cey).L(cex, cey - h * 2).C(cc2x, csy - cc2y, cc1x, csy - cc1y, csx, csy)  # Bezier curve (1st ctrlpoint,2nd control point,endpoint)

    # rootarcs[cluster]={'cluster':str(int(cluster)),'cc2x':str(cc2x),'cc2yu':str(csy+cc2y),'cc2yd':str(csy-cc2y)}
    rg.append(rpu)
    # rgc.append(rpd)
    k += 1

    clist.append(0)
    clones.append(rpu)
    # clones.append(rpd)
    # def chekIfExists(drawing: draw.Drawing, id: str):
    #    drawing.svgArgs
    # ar = 100
    pi = 0

    no_rootclusters = len(graph.get_edgelist())
    for path in reversed(allpaths):
        outdeg = 0
        moveY = 0
        for i in range(len(path)):

            # edge = edgelist.pop(0)
            # source = graph.vs.find(edge[0])
            target = graph.vs.find(path[i])

            frac = target['frac']


            # laske montako klusteria roottiin ja jaa oikeaan reunaan(eli supista k kertaa)
            cluster = target['cluster']
            #    if i in clist:

            # offy=(k-1)*20
            # clip2 = draw.ClipPath()
            # clip2.append(draw.Rectangle(0,-400,cex,cey+400)) #mask

            skewX = 0
            skewY = 0
            scaleX = 1
            scaleY = 1

            parent = graph.vs.find(cluster=target['parent'])

            vert = h / no_rootclusters

            if parent.outdegree() > 1:
                # check which of the outgoing edges this is
                moveY = parent.outdegree() * k
                # move startpoint down
            if target.outdegree() > 1:
                # siirrä alas tai ylös
                scaleY = scaleY + (outdeg * 2) / k
                outdeg = target.outdegree()
                # mp = outdeg+0.3

            # skewY=k*3
            if outdeg > 0:
                skewY = k * k
                # skewX=skewY
                translateY = -k * k
            else:
                skewY = -k * 4
                translateY = 0

            translateX = k * k

            print(path, cluster)
            h = height
            csx = x + k*10
            csy = y
            cex = x + width
            cey = h - k * k  # (h-(k+3)*20)
            cc1xu = csx + cex / 3
            cc1yu = csy + h / (k * 20)
            cc2xu = cex - k * 10
            cc2yu = cey
            cc1xl = csx + cex / 3
            cc1yl = cc1yu - k * 10
            cc2xl = cex - cex / 4
            cc2yl = csy - h + k * 10
            # print("cl",cluster)
            # print("odeg",outdeg)
            # print(pi)

            # rpu = draw.Path(fill=data.loc[data['cluster'] == cluster]['color'].values[0],fill_opacity=100.0, transform="scale("+str(scaleX)+","+str(scaleY)+") skewX("+str(skewX)+") skewY("+str(skewY)+")")
            rpu = draw.Path(fill=data.loc[data['cluster'] == cluster]['color'].values[0], fill_opacity=100.0)
            print(data.loc[data['cluster'] == cluster]['color'].values[0])

            rpu.M(csx, csy)  # Start path at point
            rpu.C(cc1xu, cc1yu, cc2xu, cc2yu, cex, cey).L(cex, csy - h + 5).C(cc2xl, cc2yl, cc1xl, cc1yl, csx, csy)
            # rootarcs[cluster]={'cluster':str(int(cluster)),'cc2x':str(cc2x),'cc2yu':str(csy+cc2y),'cc2yd':str(csy-cc2y)}
            clones.append(rpu)
            # clones.append(rpd)
            # rgi = draw.Group(id='rgi'+str(i), transform="translate("+str(translateX)+" "+str(translateY)+")")
            # rgi = draw.Group(id='rgi_'+str(cluster), transform="translate("+str(translateX)+" "+str(translateY)+")")
            rgi = draw.Group(id='rgi2_' + str(cluster))

            # rgi = draw.Group(id='rgi'+str(i))

            # print(rgi.args)

            clist.append(path[i])
            rgi.append(rpu)
            # rgi.append(rpd)
            rg.append(rgi)
            k += 1
        pi += 1

    # addAxes(rgi)

    # Save tmp image of root clones for further use to determine cluster locations

    return rg

def composeJellyBell(graph: Graph, height, width, x, y):
    allpaths = []
    dropouts = set()
    i = 0
    endvertices = set()
    for index in graph.get_adjlist():
        if index == []:
            endvertices.add(i)
            endcluster = graph.vs.find(i)['cluster']
            dropouts.add(endcluster)
            gp = graph.get_all_simple_paths(0, i, mode='all')
            if len(gp) > 0:
                allpaths.append(gp[0])
        i += 1
    # print(allpaths)

    k = 1
    root = graph.vs.find(0)
    frac = root['frac']
    cluster = root['cluster']
    # moves 2nd control points of Bezier curves downwards
    # root = draw.Arc(cx=rx, cy=ry, r=rootrad, startDeg=90, endDeg=270, fill=colors[cluster.astype(int)])

    elementids = []
    clones: draw.Path = []
    clist = []

    rootGrp = draw.Group(id='root')

    # offy=(k-1)*20
    h = height
    csx = x
    csy = y
    cex = x + width
    cey = h
    cc1x = csx + cex / 3
    cc1y = h / 10
    cc2x = cex - cex / 3
    cc2y = h + 20

    rcolor = data.loc[data['cluster'] == cluster]['color'].values[0]
    rpu = draw.Path(fill=rcolor, fill_opacity=100.0)
    rpu.M(csx, csy)  # Start path at point
    rpu.C(cc1x, csy + cc1y, cc2x, csy + cc2y, cex, cey).L(cex, cey - h * 2).C(cc2x, csy - cc2y, cc1x, csy - cc1y, csx,
                                                                              csy)  # Bezier curve (1st ctrlpoint,2nd control point,endpoint)

    # rootarcs[cluster]={'cluster':str(int(cluster)),'cc2x':str(cc2x),'cc2yu':str(csy+cc2y),'cc2yd':str(csy-cc2y)}
    rootGrp.append(rpu)
    # rgc.append(rpd)
    k += 1

    clist.append(0)
    clones.append(rpu)
    # clones.append(rpd)
    # def chekIfExists(drawing: draw.Drawing, id: str):
    #    drawing.svgArgs
    # ar = 100
    pi = 0

    no_rootclusters = len(graph.get_edgelist()) - len(endvertices) + 1

    for path in reversed(allpaths):
        outdeg = 0
        moveY = 0
        for i in range(len(path) - 1):

            # edge = edgelist.pop(0)
            # source = graph.vs.find(edge[0])

            target = graph.vs.find(path[i])

            if (path[i] in clist) == False:

                frac = target['frac']
                if target['cluster'] not in dropouts:
                    # laske montako klusteria roottiin ja jaa oikeaan reunaan(eli supista k kertaa)
                    cluster = target['cluster']
                    #    if i in clist:

                    # offy=(k-1)*20
                    # clip2 = draw.ClipPath()
                    # clip2.append(draw.Rectangle(0,-400,cex,cey+400)) #mask

                    skewX = 0
                    skewY = 0
                    scaleX = 1
                    scaleY = 1

                    parent = graph.vs.find(cluster=target['parent'])

                    vert = h / no_rootclusters

                    if parent.outdegree() > 1:
                        # check which of the outgoing edges this is
                        moveY = parent.outdegree() * k
                        # move startpoint down
                    if target.outdegree() > 1:
                        # siirrä alas tai ylös
                        scaleY = scaleY + (outdeg * 2) / k
                        outdeg = target.outdegree()
                        # mp = outdeg+0.3

                    # skewY=k*3
                    if outdeg > 0:
                        skewY = k * k
                        # skewX=skewY
                        translateY = -k * k
                    else:
                        skewY = -k * 4
                        translateY = 0

                    translateX = k * k

                    print(path, cluster)
                    csx = k * 15
                    csy = 0  # siirrä vertikaalisti jos haarautunut parentista
                    cex = x + width
                    cey = h - vert * k  # (h-(k+3)*20)
                    cc1xu = csx + cex / 3
                    cc1yu = csy + h / (k * 20)
                    cc2xu = cex - k * 10
                    cc2yu = cey
                    cc1xl = csx + cex / 3
                    cc1yl = cc1yu - k * 10
                    cc2xl = cex - cex / 4
                    cc2yl = csy - h + k * 10
                    # print("cl",cluster)
                    # print("odeg",outdeg)
                    # print(pi)

                    # rpu = draw.Path(fill=data.loc[data['cluster'] == cluster]['color'].values[0],fill_opacity=100.0, transform="scale("+str(scaleX)+","+str(scaleY)+") skewX("+str(skewX)+") skewY("+str(skewY)+")")
                    rpu = draw.Path(fill=data.loc[data['cluster'] == cluster]['color'].values[0], fill_opacity=100.0)

                    rpu.M(csx, csy)  # Start path at point
                    rpu.C(cc1xu, cc1yu, cc2xu, cc2yu, cex, cey).L(cex, csy - h + 5).C(cc2xl, cc2yl, cc1xl, cc1yl, csx, csy)  # Bezier curve (1st ctrlpoint,2nd control point,endpoint)

                    # rootarcs[cluster]={'cluster':str(int(cluster)),'cc2x':str(cc2x),'cc2yu':str(csy+cc2y),'cc2yd':str(csy-cc2y)}
                    clones.append(rpu)
                    # clones.append(rpd)
                    # rgi = draw.Group(id='rgi'+str(i), transform="translate("+str(translateX)+" "+str(translateY)+")")
                    # rgi = draw.Group(id='rgi_'+str(cluster), transform="translate("+str(translateX)+" "+str(translateY)+")")
                    rgi = draw.Group(id='rgi_' + str(cluster))

                    # rgi = draw.Group(id='rgi'+str(i))

                    # print(rgi.args)

                    clist.append(path[i])
                    rgi.append(rpu)
                    # rgi.append(rpd)
                    rootGrp.append(rgi)
                    k += 1
        pi += 1

    # addAxes(rgi)

    # Save tmp image of root clones for further use to determine cluster locations

    return rootGrp


# find models correspond to each patient from the phylogenetic tress as input
models = pd.read_csv('/home/aimaaral/dev/clonevol/data/j/mutTree_selected_models_20210311.csv', sep = '\t')
# this assembles the results from the frequency tables
# check if these are the implied models
# find files
# SAma data näyttää olevan jo aiemmassa data dataframessa
path_to_freqs = '/home/aimaaral/dev/clonevol/data/j'
files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path_to_freqs) for f in filenames if f.endswith('_cellular_freqs.csv')]

cfds = calcSampleClonalFreqs(models, files)

def drawD(scx, scy):
    ngroups = len(data.groupby("sample").groups)
    height = ngroups*200
    width = 1700
    drawing = draw.Drawing(width, height)
    frac_threshold = 0.03
    corr_treshold = 0.80


    # addAxes(d)

    # rootarcs = pd.DataFrame(data={"cluster", "arc"})

    transY = -1 * height / 2
    #transY=0
    container = draw.Group(id='container', transform="translate(0," + str(transY) + ")")
    drawing.append(container)

    # TODO: use orig data
    #cfds = data.pivot(index='sample', columns='cluster', values='frac')
    patient_cfds = cfds.filter(like=patient, axis=0)
    corrmatrix = calcCorrMatrix(patient_cfds)

    #for n in graph.dfsiter(graph.vs.find(cluster=1)):
    #    gp = graph.get_all_simple_paths(0,n.index,mode='all')
    #    if len(gp) > 0:
    #        allpaths.append(gp[0])

    branches = []

    ft = data.groupby("sample")
    # Find clusters excluded 
    dropouts = set()
    pclusters = set()
    iclusters = set()
    rclusters = set()
    masksample = set()


    for sample, corrs in corrmatrix.iterrows():
        similar = corrs.loc[corrs.index!=sample].loc[corrs > corr_treshold]
        #masksample.add(similar)
        for name in similar.index:
            #TODO: use rfind to find last index and strip the normal(DNA1 etc.) component
            cn = name[name.find("_")+1:name.rfind("_")]
            sn = sample[sample.find("_")+1:sample.rfind("_")]
            #TODO: names do not mach with original data
            # Strip patient id away
            #print("samplename:", sn)
            if cn != sn:
                #print("sname:", cn)
                masksample.add(cn)
    print("masked",masksample)
    for group_name, group in ft:
        # print(group['cluster'])
        for index, row in group.iterrows():
            inmsamples = False
            if group_name[0] == 'p':
                if row['cluster'] in pclusters:
                    inmsamples = True
                pclusters.add(row['cluster'])
            if group_name[0] == 'i':
                if row['cluster'] in iclusters:
                    inmsamples = True
                iclusters.add(row['cluster'])
            if group_name[0] == 'r':
                if row['cluster'] in rclusters:
                    inmsamples = True
                rclusters.add(row['cluster'])

            if row['parent'] in group['cluster'].tolist():
                if data.loc[data['cluster'] == row['cluster']]['frac'].max() < frac_threshold:
                    # if inmsamples == False:
                    dropouts.add(row['cluster'])  # correct but add also end vertices (done in next step)
    # If cluster is not end node but included only in interval or relapsed, exclude from root
    # If cluster is end node but in multiple samples in same treatment phase, move to root jelly
    # print(rclusters.issubset(pclusters))

    graph = build_graph(data)

    # TODO: cluster the root clones by divergence and split the JellyBell to k clusters
    #root width
    rw = 300
    rh = 200
    rootjelly = composeJellyBell(graph, rh, rw, 0, 0)
    container.append(rootjelly)
    drawing.savePng("/home/aimaaral/rootc.png")
    container.append(composeSimpleJellyBell(graph, graph.vs.find(cluster=11), graph.vs.find(cluster=17),299, 300, 400, 400))

    # edgelist = graph.get_edgelist()
    sampleboxes = {}
    tentacles = {}
    grouped_samples = data.groupby("sample")

    # box initial pos
    x = 100
    y = 150
    top = -1 * len(grouped_samples.groups) / 2 * (y + 50)
    # transY
    # TODO class object for each element so that its location and dimensions can be determined afterwards
    left = 500
    #print(grouped_samples.groups)

    gtype = "p"

    i = 0
    endvertices = set()
    allpaths = []
    for index in graph.get_adjlist():
        if index == []:
            endvertices.add(i)
            endcluster = graph.vs.find(i)['cluster']
            dropouts.add(endcluster)
            gp = graph.get_all_simple_paths(0, i, mode='all')
            if len(gp) > 0:
                allpaths.append(gp[0])
        i += 1

    drawn_clusters = []
    # TODO: group/combine(show just most presentative) the similar samples by using divergence/correlation
    for group_name, group in grouped_samples:
        #Group all elements linked to this sample
        print("Z", group_name)
        sampleGroup = draw.Group(id=group_name)
        if group_name not in masksample:
            print("gn", group_name)



            #print("##"+group_name)
            # box left pos
            if group_name[0] == "p":
                left = 500
                gtype = "p"
            if group_name[0] == "i":
                left = 700
                gtype = "i"
            if group_name[0] == "r":
                left = 700
                gtype = "r"


            top += 50
            #sample order, p,i,r
            #print(group['frac'].sum())
            gr = group.sort_values(['dfs.order'],ascending=False)
            #group['frac'].sum()
            drawnb = []
            boxjbs = []
            for index, row in gr.iterrows():

                #if top < 0:
                cluster = row['cluster']

                vertex = graph.vs.find(cluster=row['cluster'])
                frac = row['frac']
                sbheight = float(y)*float(frac)

                if cluster > -1:

                        #print(cluster)
                        if not (vertex.index in endvertices):
                            #nextv = graph.vs.find(parent=cluster)

                            outedges = vertex.out_edges()
                            for edge in outedges:
                                target = edge.target
                                tv = graph.vs.find(target)
                                if target in endvertices and tv['cluster'] in gr['cluster'].tolist():
                                    # TODO: if multiple jbs inside cluster, grow sbheught
                                    targetdata = data.loc[(data['cluster'] == tv['cluster']) & (data['sample'] == group_name)]
                                    targetfrac = targetdata['frac'].values[0]
                                    #print(tv['cluster'],parentfrac.values[0])
                                    if targetfrac > frac_threshold:

                                        if targetfrac >= frac:
                                            sbheight = targetfrac*y
                                        #Draw new jellybelly inside clone
                                        jb = draw.Path(id="jb_"+str(group_name)+"_"+str(tv['cluster']),fill=targetdata['color'].values[0],fill_opacity=100.0)
                                        csx = left+(x/2)
                                        csy=top+(sbheight/2)
                                        cex=left+x
                                        cey=csy+(sbheight/2)
                                        cc1x=csx+20
                                        cc1y=csy+5
                                        cc2x=cex-15
                                        cc2y=cey-20

                                        jb.M(csx, csy) # Start path at point
                                        jb.C(cc1x, cc1y, cc2x, cc2y, cex, cey).L(cex,csy-sbheight/2).C(cc2x, csy-(sbheight/2)+20, cc1x, csy-5, csx, csy)
                                        #drawn.append(tv['cluster'])
                                        #container.append(jb) #container foreground layer, otherwise gets hidden
                                        #ypoints = extractPointByClusterColor(clipxe-1,clipxe,0,height,data.loc[data['cluster'] == cluster]['color'].values[0],"/home/aimaaral/rootc.png")
                                        #if tv['cluster'] not in drawnt:
                                        boxjbs.append(jb)
                                        if tv['cluster'] not in drawn_clusters:
                                            drawn_clusters.append(int(tv['cluster']))
                                        # Check with H023, cluster 6 inside 2, if this indentation increased -> fixed partly

                            if frac > frac_threshold:
                                cluster = row['cluster']
                                ypoints = extractPointByClusterColor(rw-1,rw,0,height,row['color'],"/home/aimaaral/rootc.png")
                                #print(group_name, cluster, row['parent'])

                                print("sbheight",sbheight)
                                r = draw.Rectangle(left,top,x,sbheight, fill=row['color'])

                                # Draw tentacle paths
                                toff = (-1*transY)-ypoints[1]+(ypoints[1]-ypoints[0])/2
                                p = draw.Path(id="tnt"+str(cluster)+"_"+str(group_name), stroke_width=2, stroke=row['color'],fill=None,fill_opacity=0.0)

                                p.M(rw, float(toff)) # Start path at point
                                bz2ndy = top-150*frac
                                if top > 0:
                                    bz2ndy = top+150*frac

                                if gtype == "p":
                                    bz2ndx = (left-left/5)
                                if gtype == "i":
                                    bz2ndx = (left-left/3)
                                if gtype == "r":
                                    bz2ndx = (left-left/2)

                                #(rx/2+frac*rx)
                                p.C(rw+left/4, float(toff)+10, bz2ndx, bz2ndy, left, top+sbheight/2)
                                    #else:
                                    #    toff = rootarcs[idx]['rad']
                                    #    p.M(clipxe, 0+float(toff)-4)
                                #print("HERE10",group_name, cluster,sbheight,frac)
                                sampleGroup.append(r)
                                sampleGroup.append(p)

                                if cluster not in drawn_clusters:
                                    drawn_clusters.append(int(cluster))

                                    #print("HERE11",group_name, cluster)
                                    #svggr.append(draw.Text(str(cluster), 12, path=p, text_anchor='end', valign='middle'))

                            else:
                                #if row['parent'] > 0:
                                #print(row['parent'],data.loc[data['cluster'] == row['parent']]['color'].values[0])
                                    cluster = row['parent']
                                    if cluster == -1:
                                        cluster = 1
                                    parent = data.loc[(data['cluster'] == cluster) & (data['sample'] == group_name)]
                                    #print(group_name, row['cluster'], parent)
                                    frac = parent['frac'].values[0]

                                    if int(cluster) not in dropouts and sbheight > 3: # TODO: this sbheight filter is purkkafix, use parent fraction or better is to change logic so that same cluster is processed just once
                                    #if parent['frac'].values[0] > -1 : #Check orig plot case 021 why cluster 3 in iOme6 is shown when frac < 0.02
                                        # int(row['parent']) not in drawn
                                        print("sbheight2",sbheight)
                                        print("sbf",frac)
                                        ypoints = extractPointByClusterColor(rw-1,rw,0,height,parent['color'].values[0],"/home/aimaaral/rootc.png")
                                        r = draw.Rectangle(left,top,x,sbheight+3, fill=parent['color'].values[0])

                                        # Draw tentacle paths
                                        toff = (-1*transY)-ypoints[1]+(ypoints[1]-ypoints[0])/2
                                        p = draw.Path(id="tnt"+str(cluster)+"_"+str(group_name), stroke_width=2, stroke=parent['color'].values[0],fill=None,fill_opacity=0.0)

                                        p.M(rw, float(toff)) # Start path at point
                                        bz2ndy = top-150*frac
                                        if top > 0:
                                            bz2ndy = top+150*frac

                                        if gtype == "p":
                                            bz2ndx = (left-left/5)
                                        if gtype == "i":
                                            bz2ndx = (left-left/3)
                                        if gtype == "r":
                                            bz2ndx = (left-left/2)

                                        #(rx/2+frac*rx)
                                        p.C(rw+left/4, float(toff)+10, bz2ndx, bz2ndy, left, top+sbheight/2)
                                            #else:
                                            #    toff = rootarcs[idx]['rad']
                                            #    p.M(clipxe, 0+float(toff)-4)
                                        #print("HERE21", group_name, cluster,sbheight, frac)
                                        sampleGroup.append(r)
                                        sampleGroup.append(p)

                                        if cluster not in drawn_clusters:
                                            #print("HERE22", group_name, cluster)
                                            #svggr.append(draw.Text(str(cluster), 12, path=p, text_anchor='end', valign='middle'))
                                            drawn_clusters.append(int(cluster))


                                        #if jb:
                                        #    svggr.append(jb)
                            top = top+sbheight
                                    #top = top+y/ns

                                    #toff = rootarcs[i][0].args['d'].split(',')[2]
                                    #if top < 0:
                for jb in boxjbs:
                    sampleGroup.append(jb)

                #group.draw(line, hwidth=0.2, fill=colors[cc])
            label = {
            'text' : group_name,
            'fontSize' : '18',
            'fill' : 'black',
            'x' : left,
            'y':top+10,
            'startOffset': str(top),
            }
            #rg.append(draw.Use('rc', 100,100))
            sampleGroup.append(draw.Text(**label))
            sampleboxes[sampleGroup.id]=sampleGroup
            container.append(sampleGroup)

        #Draw cluster labels

        #moveSampleBox(sampleboxes['r2Asc'],-200,500)

        ci = 1
        #FIX: Use cluster
    drawn_clusters.sort(reverse=True)
    for c in drawn_clusters:

        fill = data.loc[data['cluster'] == c]['color'].values[0]
        rc = draw.Rectangle(20, 25 * ci + 100, 20, 25, fill=fill)
        dt = draw.Text(str(c), 12, x=6, y=25 * (ci + 1) + 100, valign='top')
        container.append(rc)
        container.append(dt)
        ci +=1

    return drawing


# @interact(x=300.0, y=300.0)
# def g(x,y):
#    d = drawD(x,y,scx,scy)
#    return d


# @interact(scx=1.0, scy=1.0)
# def gs(scx, scy):
#     d = drawD(scx, scy)
#     return d


d = drawD(1.0, 1.0)
d.saveSvg("./svg/" + patient + ".svg")
d.savePng("./svg/" + patient + ".png")
d