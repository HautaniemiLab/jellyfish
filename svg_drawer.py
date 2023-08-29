import drawsvg as draw
import igraph
import pandas as pd
from PIL import Image
import graph_builder
import image_processor
import image_processor as ip
import sample_analyzer


tipShape = 0.1
spreadStrength = 0.5

def get_clone_location(svgel, cluster, parent):
    for el in svgel.children:
        if isinstance(el, draw.elements.Path) == True:
            id = str(el.id)
            if id.startswith('clone'):
                idarr = id.split('_')
                print(idarr)
                if idarr[1] == str(int(cluster)) and idarr[2] == str(int(parent)):
                    print("HERE2", el.args)
                    args = el.args['d'].split(' ')
                    M = args[0].split(',')
                    C = args[1].split(',')
                    print(el.args['d'])
                    Mx = float(M[0][1:])
                    My = float(M[1])
                    return [Mx, My]


def getDepth(node):
    def fn(node):
        children = node.successors()
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


def fancystep(edge0, edge1, x):
    span = edge1 - edge0
    step = lambda x: smootherstep(edge0 - span * (1 / (1 - tipShape) - 1), edge1, x)
    atZero = step(edge0)
    return float(max(0, step(x) - atZero) / (1 - atZero))


def stackChildren(childnodes, node, spread=False):
    # print(nodes)
    # fractions = [float(n.get('fraction')) / float(node.get('fraction')) for n in node.get('children')]
    fractions = []
    for n in childnodes:
        fraction = float(n['fraction'])
        fractions.append(fraction)

    # print(node.get('children'))
    remainingSpace = float(1 - sum(fractions))

    spacing = remainingSpace / (len(fractions) + 1) if spread else 0
    cumSum = spacing if spread else remainingSpace

    positions = []
    for x in fractions:
        positions.append(cumSum + (x - 1) / 2)
        cumSum += x + spacing
    # print(positions)
    return positions


def stackTree(tree, shapers, edge=1):
    stackedNodes = dict()

    def process(node):

        top = shapers[str(node['cluster'])](edge, 0)

        bottom = shapers[str(node['cluster'])](edge, 1)
        childnodes = tree.vs.select(parent=node['cluster'])
        for child in childnodes:
            bottom = min(bottom, process(child))
        stackedNodes[str(node['cluster'])] = [top, bottom]

        return top if bottom - top > 0 else 1

    process(tree.vs.find(0))
    return stackedNodes


def lerp(a, b, x):
    return float((1 - x) * a + x * b)

def get_all_children(g, rootcluster):
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


def treeToShapers(tree: igraph.Graph, rootcluster=0):
    totalDepth = getDepth((tree.vs.find(0)))

    shapers = dict()

    def process(node, shaper, depth=0):
        if shaper is None:
            shaper = lambda x, y: y  # Make an initial shaper. Just a rectangle, no bell shape

        shapers[str(node['cluster'])] = shaper
        print(type(node))
        childnodes = tree.vs.select(parent=node['cluster'])
        spreadPositions = stackChildren(childnodes, node, True)
        stackedPositions = stackChildren(childnodes, node, False)

        #childDepth = depth + 1
        #fractionalChildDepth = childDepth / totalDepth
        childDepth = (depth + 1) if node['initialSize'] == 0 else depth
        # fractionalChildDepth = float(childDepth / totalDepth)
        fractionalChildDepth = float(childDepth / (totalDepth + totalDepth / 1000))  # purkkafix, childept==totaldepth

        def interpolateSpreadStacked(childIdx, x):
            a = smoothstep(fractionalChildDepth, 1, x)
            s = 1 - spreadStrength
            a = a * (1 - s) + s
            return lerp(spreadPositions[childIdx], stackedPositions[childIdx], a)

        for i, childNode in enumerate(childnodes):
            childFraction = childNode['fraction']
            initialSize = childNode['initialSize']

            def doInterpolateSpreadStacked(childIdx, x):
                return stackedPositions[childIdx] if initialSize > 0 else interpolateSpreadStacked(childIdx, x)

            def childShaper(x, y):
                transformedY = (
                        lerp(
                            fancystep(0 if initialSize > 0 else fractionalChildDepth, 1, x),
                            1,
                            initialSize
                        ) *
                        childFraction *
                        (y - 0.5) +
                        0.5 +
                        doInterpolateSpreadStacked(i, x)
                )
                return shaper(x, transformedY)

            process(childNode, childShaper, childDepth)

    if rootcluster != 0:
        root = tree.vs.find(cluster=rootcluster)
        pseudoRoot = root
        #pseudoRoot['initialSize'] = 1
        #pseudoRoot['fraction'] = float(1.0)
        #pseudoRoot['parent'] = 0
    else:
        ig = igraph.Graph(directed=True)
        pseudoRoot = ig.add_vertex()
        pseudoRoot['fraction'] = float(1.0)
        pseudoRoot['parent'] = 0
        pseudoRoot['cluster'] = 0
        pseudoRoot['initialSize'] = 1
        pseudoRoot['color'] = '#cccccc'
        pseudoRoot['sample'] = "pseudo"

    process(pseudoRoot, None, 0)

    return shapers


def addTreeToSvgGroup(tree, shapers, g, rootcluster=0):
    def drawNode(node):

        # Segment count. Higher number produces smoother curves.
        sc = 100

        # Find the first segment where the subclone starts to emerge
        firstSegment = 0
        for i in range(sc + 1):
            x = i / sc

            if shapers[str(node['cluster'])](x, 0) - shapers[str(node['cluster'])](x, 1) != 0:
                # The one where upper and lower edges collide
                firstSegment = max(0, i - 1)
                break

        # Start the path
        p = draw.Path(id="clone_" + str(node["cluster"]) + "_" + str(node["parent"]), fill=node["color"],
                      fill_opacity=100.0)
        p.M(firstSegment / sc, shapers[str(node['cluster'])](firstSegment / sc, 1))

        for i in range(firstSegment + 1, sc + 1):
            x = i / sc
            p.L(x, shapers[str(node['cluster'])](x, 1))

        for i in range(sc, firstSegment, -1):
            x = i / sc
            p.L(x, shapers[str(node['cluster'])](x, 0))

        g.append(p)

        childnodes = tree.vs.select(parent=node['cluster'])
        for i, childNode in enumerate(childnodes):
            drawNode(childNode)

    drawNode(tree.vs.find(0))

    return g
def addSampleToSvgGroup(tree: igraph.Graph, phase_graph: igraph.Graph, rootgraph: igraph.Graph, g, sample, rootcluster=0):
    totalDepth = getDepth(tree.vs.find(0))
    # df = tree.get_vertex_dataframe()
    # print(df[['cluster','parent','fraction']])
    # totalDepth = len(df['parent'].unique())
    # graph.get_all_shortest_paths(graph.vs.find(cluster=startcluster)
    print("totalDepth", totalDepth)

    def drawNode(node, shaper, depth=0, yh=0.0, lastfrac=0.0):
        # print(node)
        print("YH",yh, node)

        if node['cluster'] == 1:
            node['initialSize'] = 1
            sf = node.successors()[0]['frac']
            if sf > node['fraction']:
                node['fraction'] = sf

        if shaper:
            sc = 100  # Segment count. Higher number produces smoother curves.

            firstSegment = 0
            for i in range(sc + 1):
                x = i / sc

                if shaper(x, 0) - shaper(x, 1) != 0:
                    firstSegment = max(0, i - 1)
                    break

            # p = svgwrite.path.Path()



            pnode = phase_graph.vs.select(cluster=node['cluster'], sample=sample)
            childsampleclone = None
            if pnode:
                foundfromprevphase = False
                foundfromnextphase = False
                #graph_builder.getPhaseFromSampleName(sample)

                for successor in pnode[0].successors():
                    if successor['cluster'] == node['cluster']:
                        foundfromnextphase = True
                for predecessor in pnode[0].predecessors():
                    if predecessor['cluster'] == node['cluster']:
                        foundfromprevphase = True
                if not foundfromprevphase and not foundfromnextphase:
                    node['initialSize'] = 0
                    print("CLUSTER1", node) if node['cluster'] == 2 else 0
                if not foundfromprevphase and foundfromnextphase:
                    node['initialSize'] = 0
                    print("CLUSTER2", node) if node['cluster'] == 2 else 0
                if foundfromprevphase and not foundfromnextphase:
                    node['initialSize'] = 1
                    print("CLUSTER2", node) if node['cluster'] == 2 else 0
                if foundfromprevphase and foundfromnextphase:
                    node['initialSize'] = 1
                    print("CLUSTER3", node) if node['cluster'] == 2 else 0
                # never bellshape if in rootgraph
                if rootgraph.vs.select(cluster=node['cluster']):
                    node['initialSize'] = 1
                    print("CLUSTER4", node) if node['cluster'] == 10 else 0
                #childsampleclone = phase_graph.vs.select(cluster=pnode[0]['cluster'], site=pnode[0]['site'], phase=int(pnode[0]['phase'])+1)
                #print("found",pnode[0],childsampleclone[0])
            else:
                node['fraction'] = node['frac']

            height = float(node["fraction"])
            if node['initialSize'] == 0:

                p = draw.Path(id="clone_" + str(node["cluster"]) + "_" + str(node["parent"]), fill=node["color"],
                              fill_opacity=1.0)
                csx = 0.2
                csy = yh-lastfrac/2
                cex = 1
                cey = csy - lastfrac/2 + lastfrac/10
                cc1x = csx + 0.1
                cc1y = csy + lastfrac/10
                cc2x = cex - 0.1
                cc2y = cey

                p.M(csx, csy)
                p.C(cc1x, cc1y, cc2x, cc2y, cex, cey).L(cex, cey + lastfrac).C(cc2x, csy+lastfrac/2, csx-0.1, csy-lastfrac/10, csx, csy)

                # print((firstSegment / sc), yh, height, node)
                # for i in range(firstSegment + 1, sc + 1):
                #     x = i / sc
                #     p.L(x, shaper(x, 1))
                #
                # for i in range(sc, firstSegment, -1):
                #     x = i / sc
                #     p.L(x, shaper(x, 0))
                #yh = yh+lastfrac/2+height
                #yh = yh + lastfrac
            else:
                p = draw.Rectangle(firstSegment / sc, yh, 1, height, fill=node['color'], fill_opacity=1.0)

            # yh = yh + height
            g.append(p)
            print("after", node)
        else:
            shaper = lambda x, y: y  # Make an initial shaper. Just a rectangle, no bell shape

        childnodes = tree.vs.select(parent=node['cluster'])
        # childnodes = node.successors()
        # print("childnodes:",childnodes)
        spreadPositions = stackChildren(childnodes, node, True)
        stackedPositions = stackChildren(childnodes, node, False)

        childDepth = (depth + 1) if node['initialSize'] == 0 else depth
        # fractionalChildDepth = float(childDepth / totalDepth)
        fractionalChildDepth = float(childDepth / (totalDepth + totalDepth / 1000))  # purkkafix, childept==totaldepth

        def interpolateSpreadStacked(childIdx, x):
            a = smoothstep(fractionalChildDepth, 1, x)
            s = 1 - spreadStrength
            a = a * (1 - s) + s
            return lerp(spreadPositions[childIdx], stackedPositions[childIdx], a)

        # print(node['children'])
        for i, childNode in enumerate(childnodes):
            #if node['fraction'] < childNode['fraction']:
            #    node['fraction'] = childNode['fraction']
            childFraction = childNode['fraction']

            initialSize = childNode['initialSize']

            def doInterpolateSpreadStacked(childIdx, x):
                return stackedPositions[childIdx] if initialSize > 0 else interpolateSpreadStacked(childIdx, x)

            def childShaper(x, y):
                transformedY = (
                        lerp(
                            fancystep(0 if initialSize > 0 else fractionalChildDepth, 1, x),
                            1,
                            initialSize
                        ) * childFraction * (y - 0.5) + 0.5 + doInterpolateSpreadStacked(i, x)
                )
                return shaper(x, transformedY)

            drawNode(childNode, childShaper, childDepth, yh+float(node['fraction']), float(node['fraction']))

    if rootcluster != 0:
        root = tree.vs.find(cluster=rootcluster)
        pseudoRoot = root
        # pseudoRoot['initialSize'] = 1
        # pseudoRoot['fraction'] = float(1.0)
        # pseudoRoot['parent'] = 0
    else:
        ig = igraph.Graph(directed=True)
        pseudoRoot = ig.add_vertex()
        pseudoRoot['fraction'] = float(1.0)
        pseudoRoot['parent'] = 0
        pseudoRoot['cluster'] = 0
        pseudoRoot['initialSize'] = 1
        pseudoRoot['color'] = '#cccccc'
        pseudoRoot['sample'] = "pseudo"

    drawNode(pseudoRoot, None, 0)

    return g

class Drawer:
    def __init__(self, data: pd.DataFrame, graph: igraph.Graph, min_fraction, min_correlation, cfds):
        self.data = data
        self.graph = graph
        self.min_fraction = min_fraction
        self.min_correlation = min_correlation
        self.cfds = cfds

    def draw(self, scx, scy, patient):

        frac_threshold = self.min_fraction
        corr_treshold = self.min_correlation
        tipShape = 0.1
        spreadStrength = 0.5
        # TODO: use orig data
        # cfds = data.pivot(index='sample', columns='cluster', values='frac')
        patient_cfds = self.cfds.filter(like=patient, axis=0)
        # print(patient_cfds)

        corr_matrix = sample_analyzer.calc_corr_matrix(patient_cfds)

        # for n in graph.dfsiter(graph.vs.find(cluster=1)):
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
            similar = corrs.loc[corrs.index != sample].loc[corrs > corr_treshold]
            # masksample.add(similar)
            for name in similar.index:
                # TODO: use rfind to find last index and strip the normal(DNA1 etc.) component
                cn = name[name.find("_") + 1:name.rfind("_")]
                sn = sample[sample.find("_") + 1:sample.rfind("_")]
                c = 0
                if cn != sn:
                    if str(cn[1]).isnumeric():
                        if int(cn[1]) > 1:
                            masksample.add(cn)
                    else:
                        masksample.add(cn)
                    c += 1

        print("masked", masksample)
        for sample_name, group in ft:
            # print(group['cluster'])
            for index, row in group.iterrows():
                inmsamples = False
                if sample_name[0] == 'p':
                    if row['cluster'] in pclusters:
                        inmsamples = True
                    pclusters.add(row['cluster'])
                if sample_name[0] == 'i':
                    if row['cluster'] in iclusters:
                        inmsamples = True
                    iclusters.add(row['cluster'])
                if sample_name[0] == 'r':
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
        depth = len(self.data['parent'].unique())
        for index in self.graph.get_adjlist():
            if index == [] and depth > 2:
                endvertices.add(i)
                endcluster = self.graph.vs.find(i)['cluster']
                dfendc = self.data.loc[self.data['cluster'] == endcluster]
                c = 0
                for index, row in dfendc.iterrows():
                    print(row)
                    if row['frac'] > 0:
                        c += 1
                if c == 1:
                    dropouts.add(endcluster)
                gp = self.graph.get_all_simple_paths(0, i, mode='all')
                if len(gp) > 0:
                    allpaths.append(gp[0])
            i += 1
        # TODO: logic for inheriting clone from previous sample, build sample level clonal tree

        #dropouts.add(5)
        #dropouts.add(2)
        root_graph_builder = graph_builder.GraphBuilder(self.data)
        phase_graph = root_graph_builder.build_phase_graph(self.data, dropouts)

        # Check if cluster is not present in all phase 1 samples and add it to dropouts
        phasedf = phase_graph.get_vertex_dataframe().reset_index()
        phase1 = phasedf.loc[phasedf['phase'] == 1]
        phasegt1 = phasedf.loc[phasedf['phase'] > 1]
        los = []
        pgt1g = phase1.groupby("sample")
        for gname, g in pgt1g:
            print(g['cluster'])
            los.extend(g['cluster'].to_list())
            #all.add(g['cluster'])
        vcs = pd.value_counts(los)
        for vc in vcs.items():
            print("VC",vc[0],vc[1])
            if vc[1] < len(pgt1g.groups):
                dropouts.add(vc[0])

        cont = phasegt1['cluster'].isin(phase1['cluster'])
        i = 0
        for r in cont:
            if not r:
                dropouts.add(phasegt1.iloc[i]['cluster'])
            i += 1
        print("dropouts", dropouts)

        rootgraph = root_graph_builder.build_graph_sep(list(dropouts), 0, True)
        for v in rootgraph.vs:
            print("rootgraph",v)
        # TODO: cluster the root clones by divergence and split the JellyBell to k clusters
        # root width
        ngroups = len(self.data.groupby("sample").groups) - len(masksample) + 1
        height = ngroups * 250
        width = 1700
        drawing = draw.Drawing(width, height)
        ip.add_axes(drawing)

        # addAxes(d)

        rw = 250
        rh = 300

        transY = (height / 2) - rh / 2
        # transY=0
        container = draw.Group(id='container', transform="translate(0," + str(transY) + ")")
        drawing.append(container)
        # ImageProcessor.add_axes(self,container)

        rootgroup = draw.Group(id='roog', transform="scale(" + str(rw) + "," + str(rh) + ")")
        shapers = treeToShapers(rootgraph)

        rootjelly = addTreeToSvgGroup(rootgraph, shapers, rootgroup)
        container.append(rootjelly)
        tmppng = "./tmp_rootc.png"
        drawing.save_png(tmppng)
        # container.append(composeSimpleJellyBell(self.graph, self.graph.vs.find(cluster=11), self.graph.vs.find(cluster=17),299, 300, 400, 400))

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
        # print(grouped_samples.groups)

        drawn_clusters = []

        # TODO: group/combine(show just most presentative) the similar samples by using divergence/correlation
        gtype = "p"
        samplenum = 0

        self.data['phase'] = self.data['sample'].str[0:4]
        self.data['parentsample'] = "NaN"
        for index, row in self.data.iterrows():
            if row['sample'][1].isnumeric():
                s = row['sample']
                ss = ""
                for c in s:
                    if not c.isnumeric():
                        ss += c
                self.data.at[index, 'parentsample'] = ss

        phases = set(self.data['phase'].unique().tolist())
        print(self.data)
        preserved_range = [range(-1, -1)]
        left = 500

        sorted_groups = grouped_samples.groups.keys()
        sorted_groups = reversed(sorted(sorted_groups))
        drawn_tentacles = []
        for key in sorted_groups:
            # Group all elements linked to this sample
            # print("Z", group_name)

            group = grouped_samples.get_group(key)
            sample_name = key
            if sample_name not in masksample:
                # print("gn", group_name)

                # print("##"+group_name)
                # box left pos
                samplenum = str(sample_name)[1]
                if sample_name.startswith("p"):
                    left = 500
                    if samplenum.isnumeric():
                        if sample_name[0] + str(int(samplenum) - 1) in phases or self.data['phase'].str.match(
                                "^p{1}[A-Z]$") is not None:
                            if int(samplenum) > 1:
                                left = left + (int(samplenum) - 1) * 200
                                top = top - 210
                    gtype = "p"
                if sample_name.startswith("i"):
                    if "p" not in list(self.data['phase'].str[0]):
                        left = 500
                    else:
                        left = 700
                    if samplenum.isnumeric():
                        if sample_name[0] + str(int(samplenum) - 1) in phases or self.data['phase'].str.match(
                                "^i{1}[A-Z]$") is not None:
                            if int(samplenum) > 1:
                                left = left + (int(samplenum) - 1) * 200
                                top = top - 210

                    gtype = "i"
                if sample_name.startswith("r"):

                    if "i" not in list(self.data['phase'].str[0]) or "p" not in self.data['phase'].str[0]:
                        left = 700
                    else:
                        left = 900
                    if samplenum.isnumeric():
                        if sample_name[0] + str(int(samplenum) - 1) or self.data['phase'].str.match(
                                "^r{1}[A-Z]$") is not None:
                            if int(samplenum) > 1:
                                left = left + (int(samplenum) - 1) * 200
                                top = top - 210
                    gtype = "r"

                top += 100
                label = {
                    'text': sample_name,
                    'fontSize': '18',
                    'fill': 'black',
                    'x': left,
                    'y': top + 140
                }
                sampleGroup = draw.Group(id=sample_name)
                sampleGroup.append(draw.Text(**label, font_size=18))

                sample_container = draw.Group(id=sample_name,
                                              transform="translate(" + str(left) + ", " + str(top) + ") scale(" + str(
                                                  x) + "," + str(y) + ")")
                # sample order, p,i,r
                # print(group['frac'].sum())
                gr = group.sort_values(['dfs.order'], ascending=True)
                # group['frac'].sum()
                drawnb = []
                boxjbs = []
                sample_graph_builder = graph_builder.GraphBuilder(gr)
                sample_graph = sample_graph_builder.build_graph_sep_sample(list(dropouts))
                rootvertex = sample_graph.vs.find(0)
                # rootvertex['initialSize'] = 1

                samplejelly = addSampleToSvgGroup(sample_graph, phase_graph, rootgraph, sample_container, sample_name)
                #samplejelly = addSampleToSvgGroup(sample_graph, treeToShapers(sample_graph, 0), sample_container)
                container.append(samplejelly)
                drawing.save_png(tmppng)  # TODO: do in-memory
                img = Image.open(tmppng)  # Specify image path
                img_processor = ip.ImageProcessor(img)
                # TODO: Move tentacle drawing after samplegroup drawung, because now all the boxes may not be present when connecting AND use the get_el_pos_by_id()
                samplevertexes = phase_graph.vs.select(sample=sample_name)
                conn_prevphase = False
                for svertex in samplevertexes:
                    preds = svertex.predecessors()
                    for p in preds:
                        if svertex['phase'] > 1 and p['site'] == svertex['site'] and not p['sample'] == svertex['sample'] and p['phase'] <= svertex['phase']:
                            print("HERE", p['sample'], svertex['sample'], p['site'], p['cluster'])
                            conn_prevphase = True

                for index, row in gr.iterrows():
                    # if top < 0:
                    cluster = row['cluster']

                    frac = row['frac']
                    sbheight = float(y) * float(frac)

                    if cluster > -1:
                        # print(cluster)
                        # nextv = self.graph.vs.find(parent=cluster)
                        if True: #int(cluster) not in dropouts:

                            # Draw tentacle paths
                            if conn_prevphase:
                                rx = left - 101
                                ystartrange = img_processor.extract_point_by_cluster_color(rx, rx + 1, 0,
                                                                                             height,
                                                                                             row['color'])

                                starty = ystartrange[0] + (ystartrange[1] - ystartrange[0]) / 2 - transY  # (-1*transY)-ypoints[1]+(ypoints[1]-ypoints[0])/2
                                p = draw.Path(id="tnt" + str(cluster) + "_" + str(sample_name), stroke_width=2,
                                              stroke=row['color'], fill=None, fill_opacity=0.0)
                                p.M(rx, float(starty))  # Start path at point
                                yendrange = img_processor.extract_point_by_cluster_color(left + 10, left + 11,
                                                                                           int(0), int(height),
                                                                                           row['color'])
                                if yendrange[0] != 0 and yendrange[1] != 0 and [sample_name, int(cluster)] not in drawn_tentacles:
                                    preserved_range.append(range(yendrange[0], yendrange[1]))
                                    endy = yendrange[0] + (yendrange[1] - yendrange[0]) / 2 - transY
                                    print("HERE2", sample_name, rx, starty, left, endy)
                                    bz2ndy = endy
                                    bz2ndx = (left - int(samplevertexes[0]['samplenum']) * 25)
                                    p.C(rx + 25, float(starty) + 10, bz2ndx, bz2ndy, left + 1, endy)
                            else:

                                rx = rw-2
                                xy_start_pos = img_processor.get_el_pos_by_id(sample_name)
                                ystartrange = img_processor.extract_point_by_cluster_color(rx - 1, rx, xy_start_pos[1],
                                                                                             xy_start_pos[1]+150,
                                                                                             row['color'])
                                starty = ystartrange[0] + (ystartrange[1] - ystartrange[
                                    0]) / 2 - transY  # (-1*transY)-ypoints[1]+(ypoints[1]-ypoints[0])/2
                                p = draw.Path(id="tnt" + str(cluster) + "_" + str(sample_name), stroke_width=2,
                                              stroke=row['color'], fill=None, fill_opacity=0.0)
                                p.M(rx, float(starty))  # Start path at point
                                yendrange = img_processor.extract_point_by_cluster_color(left + 1, left + 2,
                                                                                           int(0), int(height),
                                                                                           row['color'])

                                if yendrange[0] != 0 and yendrange[1] != 0 or [sample_name,int(cluster)] not in drawn_tentacles:
                                    preserved_range.append(range(yendrange[0], yendrange[1]))
                                    endy = yendrange[0] + (yendrange[1] - yendrange[0]) / 2 - transY
                                    print("HERE3", sample_name, rx, starty, left, endy)

                                    bz2ndy = endy
                                    if gtype == "p":
                                        bz2ndx = (left - left / 4)
                                    if gtype == "i":
                                        bz2ndx = (left - left / 1.5)
                                    if gtype == "r":
                                        bz2ndx = (left - left / 1.5)
                                    p.C(rx + left / 4, float(starty) + 10, bz2ndx, bz2ndy, left + 1, endy)

                            sampleGroup.append(p)

                            if cluster not in drawn_clusters:
                                drawn_clusters.append(int(cluster))
                            if [sample_name, cluster] not in drawn_tentacles:
                                drawn_tentacles.append([sample_name, int(cluster)])

                        top = top + sbheight
                        # top = top+y/ns

                        # toff = rootarcs[i][0].args['d'].split(',')[2]
                        # if top < 0:

                    # group.draw(line, hwidth=0.2, fill=colors[cc])

                # rg.append(draw.Use('rc', 100,100))
                sampleboxes[sampleGroup.id] = sampleGroup
                container.append(sampleGroup)
            # Draw cluster labels

            # moveSampleBox(sampleboxes['r2Asc'],-200,500)

            ci = 1
            # FIX: Use cluster
        drawn_clusters.sort(reverse=True)

        for c in drawn_clusters:
            fill = self.data.loc[self.data['cluster'] == c]['color'].values[0]
            rc = draw.Rectangle(20, 25 * ci + 170, 20, 25, fill=fill)
            dt = draw.Text(str(c), 12, x=6, y=25 * (ci + 1) + 170, valign='top')
            container.append(rc)
            container.append(dt)
            ci += 1

        return drawing
