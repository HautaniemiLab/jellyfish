
import pathlib

import drawsvg as draw
import igraph
import pandas
from PIL import Image
from igraph import *

pandas.set_option('display.max_columns', None)


def build_graph_sep(df, rootid=0):
    def normalize_fractions(g,rootid):
        # Get the root vertex
        root = g.vs.find(rootid)

        # Recursively normalize fractions
        def normalize_vertex(vertex):
            children = vertex.successors()
            #total_fraction = sum(child['fraction'] for child in children)

            for child in children:
                print(vertex['fraction'],child['fraction'])
                if len(children) < 2:
                    child['fraction'] = vertex['fraction']-vertex['fraction']/6
                else:
                    child['fraction'] = (vertex['fraction']/len(children))
                    child['fraction'] = child['fraction']-child['fraction']/(len(children)*len(children)*len(children))   # total_fraction
                print(child['fraction'])
                normalize_vertex(child)
        normalize_vertex(root)
        return g

    graph2 = Graph(directed=True)
    dg = df.sort_values(['parent']).reset_index()
    dg = dg.groupby(["cluster","parent","color"])['frac'].sum().reset_index()

    print(dg)
    for index, row in dg.iterrows():
        parent = int(row['parent'])
        if parent == -1:
            parent = 0

        c = graph2.add_vertex()
        color = row['color']
        samples = data.loc[data['cluster']==row['cluster']]['sample']
        samples = ','.join(samples.to_list())
        c["id"] = row['cluster']
        c["label"] = row['cluster']
        c["cluster"] = int(row['cluster'])
        c["sample"] = samples
        c["fraction"] = 1.0 #/(index+1)
        c['parent'] = parent
        c["color"] = color
        c["initialSize"] = 0
        c["frac"] = row['frac']

    for index, row in dg.iterrows():
        parent = int(row['parent'])
        if parent == -1:
            parent = 0

        try:
            if parent != 0:
                i1 = graph2.vs.find(cluster=parent)
                i2 = graph2.vs.find(cluster=row['cluster'])
                #if graph.es.find(i1.index,i2.index) == False:
                graph2.add_edge(i1,i2)

        except Exception as e:
            print("Exception",e)
            #pass
    #print(graph2)
    ng = normalize_fractions(graph2, rootid)

    print("ng",ng)

    return ng

def build_graph_sep_sample(df, endvertices, rootid=0, ):
    def normalize_fractions(g,rootid):
        # Get the root vertex
        root = g.vs.find(rootid)

        # Recursively normalize fractions
        def normalize_vertex(vertex):
            children = vertex.successors()
            #total_fraction = sum(child['fraction'] for child in children)

            for child in children:
                print(vertex['fraction'],child['fraction'])
                if len(children) < 2:
                    child['fraction'] = vertex['fraction']-vertex['fraction']/6
                else:
                    child['fraction'] = (vertex['fraction']/len(children))
                    child['fraction'] = child['fraction']-child['fraction']/(len(children)*len(children)*len(children))   # total_fraction
                print(child['fraction'])
                normalize_vertex(child)
        normalize_vertex(root)
        return g

    graph2 = Graph(directed=True)
    dg = df.sort_values(['parent']).reset_index()
    #dg = dg.groupby(["cluster","parent","color"])['frac'].sum().reset_index()

    print(dg)
    for index, row in dg.iterrows():
        parent = int(row['parent'])
        if parent == -1:
            parent = 0

        c = graph2.add_vertex()
        color = row['color']
        samples = data.loc[data['cluster']==row['cluster']]['sample']
        samples = ','.join(samples.to_list())
        c["id"] = row['cluster']
        c["label"] = row['cluster']
        c["cluster"] = int(row['cluster'])
        c["sample"] = samples
        c["fraction"] = 1.0 #/(index+1)
        c['parent'] = parent
        c["color"] = color
        c["initialSize"] = 0 if row['cluster'] in endvertices else 1
        c["frac"] = row['frac']

    for index, row in dg.iterrows():
        parent = int(row['parent'])
        if parent == -1:
            parent = 0

        try:
            if parent != 0:
                i1 = graph2.vs.find(cluster=parent)
                i2 = graph2.vs.find(cluster=row['cluster'])
                #if graph.es.find(i1.index,i2.index) == False:
                graph2.add_edge(i1,i2)

        except Exception as e:
            print("Exception",e)
            #pass
    #print(graph2)
    ng = normalize_fractions(graph2, rootid)

    print("ng",ng)

    return ng

class GraphBuilder:
    def __init__(self, patientdf):
        self.patientdf = patientdf

    def build_graph(self):
        graph = Graph(directed=True)
        dg = self.patientdf.sort_values(['parent']).reset_index()
        dg = dg.groupby(["cluster","parent","color"])['frac'].sum().reset_index()

        ndf = dg[["cluster","parent","frac"]].copy().sort_values(['parent']).reset_index()
        ndf['frac'] = ndf['frac']/ndf['frac']

        newfrfrac = 1.0
        for index, row in dg.iterrows():
            parent = int(row['parent'])
            if parent == -1:
                parent = 0

            parents = ndf.loc[ndf["parent"]==row["parent"]]
            pcount = len(parents)
            if parent == 0:
                newfrac = 1.0
            else:
                p = ndf.loc[ndf["cluster"]==int(row["parent"])]
                # idx = ndf.loc[ndf["cluster"]==row["cluster"]].index.values[0]
                pfrac = ndf.at[p.index.values[0],'frac']
                #print(p)
                #print("fr:",str(p['frac']),"pc",str(pcount))
                newfrac = float(pfrac/pcount)
                idx = ndf.loc[ndf["cluster"]==row["cluster"]].index.values[0]
                ndf.at[idx,'frac'] = newfrac

            c = graph.add_vertex()
            color = "#cccccc"
            if parent != 0:
                color = row['color']
            samples = data.loc[data['cluster']==row['cluster']]['sample']
            samples = ','.join(samples.to_list())
            c["id"] = row['cluster']
            c["label"] = row['cluster']
            c["cluster"] = int(row['cluster'])
            c["sample"] = samples
            c["fraction"] = (newfrac - 0.02)
            c['parent'] = parent
            c["color"] = color
            c["initialSize"] = 0
            c["frac"] = row['frac']

        print(ndf.sort_values(['parent']))

        for index, row in dg.iterrows():
            parent = int(row['parent'])
            if parent == -1:
                parent = 0

            try:
                if parent != 0:
                    i1 = graph.vs.find(cluster=parent)
                    i2 = graph.vs.find(cluster=row['cluster'])
                    #if graph.es.find(i1.index,i2.index) == False:
                    if (i1.index,i2.index) not in graph.get_edgelist()[0:]:
                        graph.add_edge(i1,i2)

            except Exception as e:
                pass

        return graph



    def build_graph2(self, rootid=0):
        def normalize_fractions(g,rootid):
            # Get the root vertex
            root = g.vs.find(rootid)

            # Recursively normalize fractions
            def normalize_vertex(vertex):
                children = vertex.successors()
                #total_fraction = sum(child['fraction'] for child in children)
                print(vertex['fraction'])
                for child in children:
                    if len(children) < 2:
                        child['fraction'] = vertex['fraction'] - 0.2
                    else:
                        child['fraction'] = (vertex['fraction']/len(children)) - 0.02 # total_fraction
                    normalize_vertex(child)
            normalize_vertex(root)
            return g

        graph2 = Graph(directed=True)
        dg = self.patientdf.sort_values(['parent']).reset_index()

        for index, row in dg.iterrows():
            parent = int(row['parent'])
            if parent == -1:
                parent = 0

            c = graph2.add_vertex()
            color = row['color']
            samples = data.loc[data['cluster']==row['cluster']]['sample']
            samples = ','.join(samples.to_list())
            c["id"] = row['cluster']
            c["label"] = row['cluster']
            c["cluster"] = int(row['cluster'])
            c["sample"] = samples
            c["fraction"] = 1.0 #/(index+1)
            c['parent'] = parent
            c["color"] = color
            c["initialSize"] = 0
            c["frac"] = row['frac']

        for index, row in dg.iterrows():
            parent = int(row['parent'])
            if parent == -1:
                parent = 0

            try:
                if parent != 0:
                    i1 = graph2.vs.find(cluster=parent)
                    i2 = graph2.vs.find(cluster=row['cluster'])
                    #if graph.es.find(i1.index,i2.index) == False:
                    if (i1.index,i2.index) not in graph2.get_edgelist()[0:]:
                        graph2.add_edge(i1,i2)

            except Exception as e:
                print("Exception",e)
                pass
        print(graph2)

        #ng = normalize_fractions(graph2, rootid)

        #print("ng",ng)

        return graph2


    def build_graph_per_sample(self, endvertices):

        graph = Graph(directed=True)

        for index, row in self.patientdf.iterrows():

            parent = int(row['parent'])
            if parent == -1:
                parent = 0

            c= graph.add_vertex()
            color = "#cccccc"
            if parent != 0:
                color = row['color']

            c["id"] = row['cluster'] if parent != 0 else 0
            c["label"] = row['cluster']
            c["cluster"] = int(row['cluster'])
            c["sample"] = row['sample']
            c["fraction"] = row['frac']
            c['parent'] = parent
            c["color"] = color
            c["initialSize"] = 0 if row['cluster'] in endvertices else 1


        for index, row in self.patientdf.iterrows():
            parent = int(row['parent'])
            if parent == -1:
                parent = 0

            try:
                if parent != 0:
                    i1 = graph.vs.find(cluster=parent, sample=row['sample'])
                    i2 = graph.vs.find(cluster=row['cluster'], sample=row['sample'])
                    #if graph.es.find(i1.index,i2.index) == False:
                    if (i1.index,i2.index) not in graph.get_edgelist()[0:]:
                        graph.add_edge(i1,i2)

            except Exception as e:
                pass

        return graph

def get_clone_location(svgel, cluster, parent):
    for el in svgel.children:
        if isinstance(el, draw.elements.Path) == True:
            id = str(el.id)
            if id.startswith('clone'):
                idarr = id.split('_')
                print(idarr)
                if idarr[1]==str(int(cluster)) and idarr[2]==str(int(parent)):
                    print("HERE2",el.args)
                    args = el.args['d'].split(' ')
                    M = args[0].split(',')
                    C = args[1].split(',')
                    print(el.args['d'])
                    Mx = float(M[0][1:])
                    My = float(M[1])
                    return [Mx,My]




class ImageProcessor:
    def __init__(self, image):
        self.image = image

    def removeRootCluster(self):
        for el in self.group.allChildren():
            if isinstance(el, draw.elements.Path) == True:
                if str(el.id).startswith('rgi'):
                    print(el)


    def moveSampleBox(self, moveX, moveY):
        self.group.args['transform'] = 'translate(' + str(moveX) + ',' + str(moveY) + ')'
        for el in self.group.all_children():
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

    def extract_point_by_cluster_color(self, sx, ex, sy, ey, color):
        pixels = self.image.load()
        width, height = self.image.size

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

    def add_axes(self, el):
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
            el.append(draw.Text(**te,font_size=12))

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
            el.append(draw.Text(**ty,font_size=12))

import pandas as pd
import os
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt

class DataAnalyzer:
    def __init__(self, models, files):
        self.models = models
        self.files = files

    def calc_sample_clonal_freqs(self, freqs: pandas.DataFrame):

        unique_samples = freqs['sample'].unique()
        unique_clones = np.arange(1, freqs['cluster'].max()+1)
        values = np.zeros((len(unique_clones), len(unique_samples)))
        rownames = unique_clones.astype(str)
        colnames = unique_samples

        for i, clone in enumerate(unique_clones):
            for j, sample in enumerate(unique_samples):
                clone_sample_freqs = freqs.loc[(freqs['cluster'] == clone) & (freqs['sample'] == sample), 'freq']
                if not clone_sample_freqs.empty:
                    values[i, j] = clone_sample_freqs.iloc[0]
        assert np.all((0 <= values) & (values <= 1.))
        assert np.all((1-.05 <= np.sum(values, axis = 0)) & (np.sum(values, axis = 0) <= 1+.05))

        # Calculate Kullback-Leibler divergence of the sample’s clonal frequency distribution from the average distribution over all samples of a patientid
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

        aug = pd.DataFrame(p, columns=[f'w{i}' for i in range(1, p.shape[1]+1)], index = unique_samples)
        aug.index.names=['sample']
        # save averaged clonal frequency distributions per sample
        # TODO: add equation coefficients (hc etc) columns and join dataframes (wi means ith clone and value is 𝑝𝑖𝑗 is the normalized frequency)
        #intres.append(pd.DataFrame({'hc': hc, 'c': c, 'hu': hu, 'u': u, 'n': n}, index = unique_samples, columns = aug.columns))

        #results = pd.concat(intres, axis=0, ignore_index=False)
        #auf.to_csv("/home/aimaaral/dev/tumor-evolution-2023/heterogeneity/avg_cfd.csv",sep = '\t')
        return aug.fillna(0)

    def calc_all_clonal_freqs(self):
        intres = []
        augarr = []
        for file in self.files:
            print(f"processing '{file}'..")
            basename = os.path.basename(file)
            patientid = re.sub(f'^([^_]+\\d+)(_v2)?_vaf_(.*)_cellular_freqs\\.csv$', '\\1', basename)

            freqs = pd.read_csv(file, sep = '\t') # non utf start byte error encoding='ISO-8859-1'
            freqs = freqs.loc[freqs['model.num'] == models.loc[models['patient'].str.contains(patientid), 'model'].values[0], :]

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

            # Calculate Kullback-Leibler divergence of the sample’s clonal frequency distribution from the average distribution over all samples of a patientid
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

            aug = pd.DataFrame(p, columns=[f'w{i}' for i in range(1, p.shape[1]+1)], index = unique_samples)
            aug.index.names=['sample']
            augarr.append(aug)
            # save averaged clonal frequency distributions per sample
            # TODO: add equation coefficients (hc etc) columns and join dataframes (wi means ith clone and value is 𝑝𝑖𝑗 is the normalized frequency)
            #intres.append(pd.DataFrame({'hc': hc, 'c': c, 'hu': hu, 'u': u, 'n': n}, index = unique_samples, columns = aug.columns))
        auf = pd.concat(augarr, axis=0, ignore_index=False).fillna(0)
        #results = pd.concat(intres, axis=0, ignore_index=False)
        #auf.to_csv("/home/aimaaral/dev/tumor-evolution-2023/heterogeneity/avg_cfd.csv",sep = '\t')
        return auf

    def calc_corr_matrix(self, cell_freqs):
        sns.set_theme(style="white")

        # Generate a large random dataset
        #d = pd.read_csv('/home/aimaaral/dev/tumor-evolution-2023/heterogeneity/cellular_freqs.tsv',sep = '\t').set_index("sample").drop(columns=["hc","hu","c","u","n"])

        # TODO: group by patient and generate corrmatrix for each patient separately
        # Compute the correlation matrix
        corr = cell_freqs.T.corr()
        #print(d.T)

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 10, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        try:
            hmap = sns.heatmap(corr, cmap=cmap, center=0,
                               square=True, linewidths=.5, cbar_kws={"shrink": .5})
            hmap
        except Exception as e:
            print(e)
            pass
        return corr


def getDepth(node):
    def fn(node):
        children = node.get('children', [])
        depths = []
        for child in children:
            depth = fn(child)
            depths.append(depth)
        max_depth = max(depths) if depths else 0
        return max_depth + 1

    return fn(node)

def clamp(lower, upper, x):
    return max(lower, min(upper, x))

def smoothstep(edge0, edge1, x):
    x = clamp(0, 1, (x - edge0) / (edge1 - edge0))
    return float(x * x * (3 - 2 * x))

def smootherstep(edge0, edge1, x):
    x = clamp(0, 1, (x - edge0) / (edge1 - edge0))
    return float(x * x * x * (3.0 * x * (2.0 * x - 5.0) + 10.0))

def fancystep(edge0, edge1, x, tipShape):
    span = edge1 - edge0
    step = lambda x: smootherstep(edge0 - span * (1 / (1 - tipShape) - 1), edge1, x)
    atZero = step(edge0)
    return float(max(0, step(x) - atZero) / (1 - atZero))

def stackChildren(nodes, node, spread=False):
    #print(nodes)
    #fractions = [float(n.get('fraction')) / float(node.get('fraction')) for n in node.get('children')]
    fractions = []
    for n in nodes:
        if node['fraction'] == 0.0:
            node['fraction'] = 1.0
        print(float(n['fraction']),float(node['fraction']))
        fraction = float(n['fraction']) / float(node['fraction'])
        fractions.append(fraction)

    #print(node.get('children'))
    remainingSpace = float(1 - sum(fractions))

    spacing = remainingSpace / (len(fractions) + 1) if spread else 0
    cumSum = spacing if spread else remainingSpace

    positions = []
    for x in fractions:
        positions.append(cumSum + (x - 1) / 2)
        cumSum += x + spacing
    #print(positions)
    return positions

def lerp(a, b, x):
    return float((1 - x) * a + x * b)

tipShape = 0.1
spreadStrength = 0.5

def get_all_children(g,rootcluster):
    # Get the root vertex
    root = g.vs.find(cluster=rootcluster)
    # Recursively normalize fractions
    childrenids = set()
    def get_children(vertex):
        children = vertex.successors()
        print(children)
        for child in children:
            childrenids.add(child.index)
            get_children(child)
    get_children(root)
    return list(childrenids)

def addTreeToSvgGroup(tree: Graph, g, tipShape, spreadStrength, rootid = 0):
    #totalDepth = getDepth(tree)
    df = tree.get_vertex_dataframe()
    #print(df[['cluster','parent','fraction']])
    totalDepth = len(df['parent'].unique())
    #graph.get_all_shortest_paths(graph.vs.find(cluster=startcluster)
    print("totalDepth",totalDepth)

    def drawNode(node, shaper, depth=0):
        #print(node)
        if shaper:
            sc = 100  # Segment count. Higher number produces smoother curves.

            firstSegment = 0
            for i in range(sc + 1):
                x = i / sc

                if shaper(x, 0) - shaper(x, 1) != 0:
                    firstSegment = max(0, i - 1)
                    break

            #p = svgwrite.path.Path()
            p = draw.Path(id="clone_"+str(node["cluster"])+"_"+str(node["parent"]),fill=node["color"] ,fill_opacity=100.0)
            p.M(firstSegment / sc, shaper(firstSegment / sc, 1))

            for i in range(firstSegment + 1, sc + 1):
                x = i / sc
                p.L(x, shaper(x, 1))

            for i in range(sc, firstSegment, -1):
                x = i / sc
                p.L(x, shaper(x, 0))

            g.append(p)

        else:
            shaper = lambda x, y: y  # Make an initial shaper. Just a rectangle, no bell shape

        childnodes = tree.vs.select(parent=node['cluster'])
        #childnodes = node.successors()
        #print("childnodes:",childnodes)
        spreadPositions = stackChildren(childnodes, node, True)
        stackedPositions = stackChildren(childnodes, node, False)

        childDepth = depth + 1
        fractionalChildDepth = float(childDepth / totalDepth)

        def interpolateSpreadStacked(childIdx, x):
            a = smoothstep(fractionalChildDepth, 1, x)
            s = 1 - spreadStrength
            a = a * (1 - s) + s
            return lerp(spreadPositions[childIdx], stackedPositions[childIdx], a)

        #print(node['children'])
        for i, childNode in enumerate(childnodes):
            childFraction = childNode['fraction'] / node['fraction']
            initialSize = childNode['initialSize']

            def doInterpolateSpreadStacked(childIdx, x):
                return stackedPositions[childIdx] if initialSize > 0 else interpolateSpreadStacked(childIdx, x)

            def childShaper(x, y):
                transformedY = (
                        lerp(
                            fancystep(0 if initialSize > 0 else fractionalChildDepth, 1, x, tipShape),
                            1,
                            initialSize
                        ) * childFraction * (y - 0.5) + 0.5 + doInterpolateSpreadStacked(i, x)
                )
                return shaper(x, transformedY)

            drawNode(childNode, childShaper, childDepth)

    if rootid != 0:
        root = tree.vs.find(cluster=rootid)
        pseudoRoot = dict(fraction = float(1.0), parent = 0, cluster = rootid, color=root['color'], sample=root['sample'])
    else:
        pseudoRoot = dict(fraction = float(1.0), parent = 0, cluster = 0, color='#cccccc', sample="pseudo")
    #pseudoRoot = tree.add_vertex(fraction = float(1.0), parent = 0, cluster = 1, color="#cccccc", sample="pseudo")
    #drawNode(tree.vs.find(parent=0), lambda x, y: y, 0)
    drawNode(pseudoRoot, None, 0)

    return g


class Drawer:
    def __init__(self, data: pandas.DataFrame, graph: igraph.Graph, min_fraction, min_correlation):
        self.data = data
        self.graph = graph
        self.min_fraction = min_fraction
        self.min_correlation = min_correlation

    def draw(self, scx, scy, patient):

        frac_threshold = self.min_fraction
        corr_treshold = self.min_correlation
        tipShape = 0.1
        spreadStrength = 0.5


        # TODO: use orig data
        #cfds = data.pivot(index='sample', columns='cluster', values='frac')
        patient_cfds = cfds.filter(like=patient, axis=0)
        #print(patient_cfds)
        corr_matrix = data_analyzer.calc_corr_matrix(patient_cfds)

        #for n in graph.dfsiter(graph.vs.find(cluster=1)):
        #    gp = graph.get_all_simple_paths(0,n.index,mode='all')
        #    if len(gp) > 0:
        #        allpaths.append(gp[0])

        branches = []

        ft = self.data.groupby("sample")
        # Find clusters excluded
        dropouts = set()
        pclusters = set()
        iclusters = set()
        rclusters = set()
        masksample = set()

        for sample, corrs in corr_matrix.iterrows():
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
                    if self.data.loc[self.data['cluster'] == row['cluster']]['frac'].max() < frac_threshold:
                        # if inmsamples == False:
                        dropouts.add(row['cluster'])  # correct but add also end vertices (done in next step)
        # If cluster is not end node but included only in interval or relapsed, exclude from root
        # If cluster is end node but in multiple samples in same treatment phase, move to root jelly
        # print(rclusters.issubset(pclusters))
        i = 0
        endvertices = set()
        allpaths = []

        for index in self.graph.get_adjlist():
            if index == []:
                endvertices.add(i)
                endcluster = self.graph.vs.find(i)['cluster']
                dropouts.add(endcluster)
                gp = self.graph.get_all_simple_paths(0, i, mode='all')
                if len(gp) > 0:
                    allpaths.append(gp[0])
            i += 1

        # TODO: cluster the root clones by divergence and split the JellyBell to k clusters
        #root width
        ngroups = len(self.data.groupby("sample").groups) - len(masksample) +1
        height = ngroups*200
        width = 1700
        drawing = draw.Drawing(width, height)
        ImageProcessor.add_axes(self,drawing)

        # addAxes(d)

        rw = 250
        rh = 300

        transY = (height/2)-rh/2
        #transY=0
        container = draw.Group(id='container', transform="translate(0," + str(transY) + ")")
        drawing.append(container)
        #ImageProcessor.add_axes(self,container)

        communities = graph.community_edge_betweenness()
        communities = communities.as_clustering()
        # TODO: get parent of communitys root and all outgoing paths from it,  add path vertices to subgraph

        #for i, community in enumerate(communities):

        i = 0
        # TODO: create logic to find the splitting clusters
        for c in {6}:
            #print(community)
            #community_graph = communities.subgraph(i)

            treeids = get_all_children(graph, c)
            first = graph.vs.find(cluster=c)

            #children = first.successors()
            print("coo",treeids)
            #op = graph.spanning_tree(first)
            #print(op)

            #preds = None
            #try:
            #    preds = graph.vs.find(community_graph.vs.find(0)).predecessors()
            #except:
            #    pass

            #if preds:
                #print("preds",preds)
                #community = [preds[0].index]+community
            subgraph = graph.induced_subgraph(treeids)
            #print("comm",community)

            print("sub",subgraph)
            igraph.plot(subgraph, "./g"+str(i)+".pdf")
            #df = subgraph.get_vertex_dataframe().reset_index()
            #print(df)
            #cgc = build_graph_sep(df)
            #print(cgc)
            #cg = draw.Group(id='cg'+str(i), transform="translate(300, "+str(i*200)+") scale("+str(rw)+","+str(rh)+")")

            #cgsvg = addTreeToSvgGroup(subgraph, cg, tipShape, spreadStrength, subgraph.vs.find(0)['cluster'])
            #container.append(cgsvg)
            i=1
            #print(community_graph.get_vertex_dataframe())
            #print(community_graph.get_edge_dataframe())
            #rootdata = preprocessBellClonesGraph(community_graph.get_vertex_dataframe(), [])
            #print(community_graph.vs['parent'])
            #print(preprocessBellClonesGraph(community_graph.get_vertex_dataframe(), []))
            #community_edges = graph.es.select(_within=community)


        rootgroup = draw.Group(id='roog', transform="scale("+str(rw)+","+str(rh)+")")
        rootjelly = addTreeToSvgGroup(graph, rootgroup, tipShape, spreadStrength)
        container.append(rootjelly)
        tmppng = "./tmp_rootc.png"
        drawing.save_png(tmppng)
        #container.append(composeSimpleJellyBell(self.graph, self.graph.vs.find(cluster=11), self.graph.vs.find(cluster=17),299, 300, 400, 400))

        # edgelist = self.graph.get_edgelist()
        sampleboxes = {}
        tentacles = {}
        grouped_samples = self.data.groupby("sample")

        # box initial size
        x = 100
        y = 150
        top = (-height / 4)
        # transY
        # TODO class object for each element so that its location and dimensions can be determined afterwards
        left = 500
        #print(grouped_samples.groups)

        drawn_clusters = []

        img = Image.open(tmppng)  # Specify image path
        image_processor = ImageProcessor(img)
        # TODO: group/combine(show just most presentative) the similar samples by using divergence/correlation
        gtype = "p"

        data['phase'] = data['sample'].str[0]
        phases = set(data['phase'].unique().tolist())

        for group_name, group in grouped_samples:
            #Group all elements linked to this sample
            #print("Z", group_name)


            if group_name not in masksample:
                #print("gn", group_name)

                #print("##"+group_name)
                # box left pos

                if group['phase'].values[0] == "p":
                    left = 500
                    gtype = "p"
                if group['phase'].values[0] == "i":
                    left = 700
                    gtype = "i"
                if group['phase'].values[0] == "r":
                    if "i" not in phases:
                        left = 700
                    else:
                        left = 900
                    gtype = "r"

                top += 50

                label = {
                    'text' : group_name,
                    'fontSize' : '18',
                    'fill' : 'black',
                    'x' : left,
                    'y':top+20
                }
                sampleGroup = draw.Group(id=group_name)
                sampleGroup.append(draw.Text(**label, font_size=18))
                sample_container = draw.Group(id=group_name, transform="translate("+str(left)+", "+str(top)+") scale("+str(x)+","+str(y)+")")
                #sample order, p,i,r
                #print(group['frac'].sum())
                gr = group.sort_values(['dfs.order'], ascending=True)
                #group['frac'].sum()
                drawnb = []
                boxjbs = []


                sample_graph = build_graph_sep_sample(gr, endvertices)
                rootvertex = sample_graph.vs.find(0)
                #rootvertex['initialSize'] = 1

                samplejelly = addTreeToSvgGroup(sample_graph, sample_container, tipShape, spreadStrength, rootvertex['cluster'])

                for index, row in gr.iterrows():

                    #if top < 0:
                    cluster = row['cluster']

                    vertex = self.graph.vs.find(cluster=row['cluster'])
                    frac = row['frac']
                    sbheight = float(y)*float(frac)

                    if cluster > -1:

                        #print(cluster)
                        if not (vertex.index in endvertices):
                            #nextv = self.graph.vs.find(parent=cluster)

                            outedges = vertex.out_edges()
                            for edge in outedges:
                                target = edge.target
                                tv = self.graph.vs.find(target)

                                #path_to_end = self.graph.get_all_simple_paths(edge.target, mode='in')
                                #self.graph.es.find(target)

                                if target in endvertices and tv['cluster'] in gr['cluster'].tolist():
                                #if tv['cluster'] in gr['cluster'].tolist():

                                    # TODO: if multiple jbs inside cluster, combine to new jellybell starting from parent (check H032)
                                    targetdata = self.data.loc[(self.data['cluster'] == tv['cluster']) & (self.data['sample'] == group_name)]
                                    targetfrac = targetdata['frac'].values[0]
                                    #print(tv['cluster'],parentfrac.values[0])
                                    if targetfrac > frac_threshold:

                                        if targetfrac >= frac:
                                            sbheight = targetfrac*y
                                        #jb = JellyBellComposer.compose_simple_jelly_bell(data, graph, sbheight, x, left, top, vertex.index, tv.index)
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

                                        #boxjbs.append(jb)
                                        if tv['cluster'] not in drawn_clusters:
                                            drawn_clusters.append(int(tv['cluster']))
                                        # Check with H023, cluster 6 inside 2, if this indentation increased -> fixed partly

                            if frac > frac_threshold:
                                cluster = row['cluster']
                                ypoints = image_processor.extract_point_by_cluster_color(rw - 1, rw, 0, height, row['color'])
                                #print(group_name, cluster, row['parent'])

                                r = draw.Rectangle(left,top,x,sbheight, fill=row['color'])

                                # Draw tentacle paths
                                toff = ypoints[0]+(ypoints[1]-ypoints[0])/2-transY #(-1*transY)-ypoints[1]+(ypoints[1]-ypoints[0])/2
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

                                #sampleGroup.append(r)
                                sampleGroup.append(p)

                                if cluster not in drawn_clusters:
                                    drawn_clusters.append(int(cluster))

                                    #print("HERE11",group_name, cluster)
                                    #svggr.append(draw.Text(str(cluster), 12, path=p, text_anchor='end', valign='middle'))

                            else:
                                #if row['parent'] > 0:
                                #print(row['parent'],self.data.loc[self.data['cluster'] == row['parent']]['color'].values[0])
                                cluster = row['parent']
                                if cluster == -1:
                                    cluster = 1
                                parent = self.data.loc[(self.data['cluster'] == cluster) & (self.data['sample'] == group_name)]
                                #print(group_name, row['cluster'], parent)
                                frac = parent['frac'].values[0]

                                if int(cluster) not in dropouts and sbheight > 3: # TODO: this sbheight filter is purkkafix, use parent fraction or better is to change logic so that same cluster is processed just once

                                    ypoints = image_processor.extract_point_by_cluster_color(rw - 1, rw, 0, height, parent['color'].values[0])
                                    r = draw.Rectangle(left,top,x,sbheight+3, fill=parent['color'].values[0])

                                    # Draw tentacle paths
                                    toff = ypoints[0]+(ypoints[1]-ypoints[0])/2 - transY # (-1*transY)-ypoints[1]+(ypoints[1]-ypoints[0])/2
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

                                    #sampleGroup.append(r)
                                    sampleGroup.append(p)

                                    if cluster not in drawn_clusters:
                                        #print("HERE22", group_name, cluster)
                                        #svggr.append(draw.Text(str(cluster), 12, path=p, text_anchor='end', valign='middle'))
                                        drawn_clusters.append(int(cluster))


                            top = top+sbheight
                            #top = top+y/ns

                            #toff = rootarcs[i][0].args['d'].split(',')[2]
                            #if top < 0:
                    for jb in boxjbs:
                        sampleGroup.append(jb)

                    #group.draw(line, hwidth=0.2, fill=colors[cc])

                #rg.append(draw.Use('rc', 100,100))

                sampleboxes[sampleGroup.id]=sampleGroup
                container.append(sampleGroup)
                container.append(samplejelly)

            #Draw cluster labels

            #moveSampleBox(sampleboxes['r2Asc'],-200,500)

            ci = 1
            #FIX: Use cluster
        drawn_clusters.sort(reverse=True)
        for c in drawn_clusters:

            fill = self.data.loc[self.data['cluster'] == c]['color'].values[0]
            rc = draw.Rectangle(20, 25 * ci + 170, 20, 25, fill=fill)
            dt = draw.Text(str(c), 12, x=6, y=25 * (ci + 1) + 170, valign='top')
            container.append(rc)
            container.append(dt)
            ci +=1

        return drawing

if __name__ == "__main__":
    clonevol_preproc_data_path = "/Users/aimaaral/dev/clonevol/data/preproc/"
    #clonevol_freq_data_path = "/Users/aimaaral/dev/clonevol/data/j/"
    clonevol_freq_data_path = "/Users/aimaaral/dev/clonevol/data/cellular_freqs/"
    mut_trees_file = "/Users/aimaaral/dev/clonevol/data/j/mutTree_selected_models_20210311.csv"
    models = pd.read_csv(mut_trees_file, sep = '\t')
    files = list(pathlib.Path(clonevol_freq_data_path).rglob("*[0-9]_cellular_freqs.csv"))
    #files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(clonevol_freq_data_path) for f in filenames if f.endswith('_cellular_freqs.csv')]

    data_analyzer = DataAnalyzer(models, files)
    cfds = data_analyzer.calc_all_clonal_freqs()
    #preproc_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(clonevol_preproc_data_path) for f in filenames if f.endswith('.csv')]
    preproc_files = ["/Users/aimaaral/dev/clonevol/data/preproc/H030.csv"]
    for patientcsv in preproc_files:
        fnsplit = patientcsv.split('/')
        patient = fnsplit[len(fnsplit)-1].split('.')[0]
        data = pd.read_csv(patientcsv, sep=",")
        data = data.drop(data.columns[0], axis=1).dropna(axis='rows')
        print(data)
        # "/Users/aimaaral/dev/clonevol/examples/" + patient + ".csv", sep=","
        #graph_builder = GraphBuilder(data)
        graph = build_graph_sep(data)
        drawer = Drawer(data, graph, 0.000001, 0.999999)
        jellyplot = drawer.draw(1.0, 1.0, patient)
        jellyplot.save_svg("./svg/" + patient + ".svg")
        jellyplot.save_png("./png/" + patient + ".png")
