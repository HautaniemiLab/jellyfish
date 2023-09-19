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


def get_el_pos_by_id(svggroup: draw.Group, context, id):
    for el in svggroup.all_children(context):
        # if isinstance(el, draw.elements.Path) == True:
        #     if str(el.id) == id:
        #         print(el.id, id)
        #         args = el.args['d'].split(' ')
        #         M = args[0].split(',')
        #         C = args[1].split(',')
        #         print(el.args['d'])
        #         Mx = float(M[0][1:])
        #         My = float(M[1])
        #
        #         return [Mx, My]

        if isinstance(el, draw.elements.Rectangle) == True:
            if str(el.id) == id:
                print(el.id, id)
                print(el.args)
                x = float(el.args['x'])
                y = float(el.args['y'])
                w = float(el.args['width'])
                h = float(el.args['height'])
                t = el.args['translate']
                s = el.args['scale']
                return [x+float(t[0]), y+float(t[1]), w*float(s[0]), h*float(s[1])]

def get_el_pos_of_group(el: draw.Group):
    x = float(el.args['x'])
    y = float(el.args['y'])

    return [x, y]

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
    at_zero = step(edge0)
    return float(max(0, step(x) - at_zero) / (1 - at_zero))


def stack_children(childnodes, node, spread=False):
    # print(nodes)
    # fractions = [float(n.get('fraction')) / float(node.get('fraction')) for n in node.get('children')]
    fractions = []
    for n in childnodes:
        fraction = float(n['fraction'])
        fractions.append(fraction)

    # print(node.get('children'))
    remaining_space = float(1-sum(fractions))

    spacing = remaining_space / (len(fractions) + 1) if spread else 0
    cum_sum = spacing if spread else remaining_space

    positions = []
    for x in fractions:
        positions.append(cum_sum + (x - 1) / 2)
        cum_sum += x + spacing
    # print(positions)
    return positions

def stackChildrenV1(nodes, node, spread=False):
    # print(nodes)
    # fractions = [float(n.get('fraction')) / float(node.get('fraction')) for n in node.get('children')]
    fractions = []
    for n in nodes:
        #if node['fraction'] == 0.0:
        #    node['fraction'] = 1.0
        fraction = float(n['fraction']) #/ float(node['fraction'])
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


def stackTree(tree: igraph.Graph, shapers: dict, edge=1):
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


def tree_to_shapers(tree: igraph.Graph, rootcluster=0):
    total_depth = getDepth((tree.vs.find(0)))

    shapers = dict()

    def process(node, shaper, depth=0):

        if shaper is None:
            shaper = lambda x, y: y  # Make an initial shaper. Just a rectangle, no bell shape

        shapers[str(node['cluster'])] = shaper

        childnodes = tree.vs.select(parent=node['cluster'])
        #reduce_frac = 1/len(tree.vs)
        spread_positions = stack_children(childnodes, node, True)
        stacked_positions = stack_children(childnodes, node, False)

        #childDepth = depth + 1
        #fractionalChildDepth = childDepth / totalDepth
        childDepth = (depth+1) if node['initialSize'] == 0 else depth
        # fractionalChildDepth = float(childDepth / totalDepth)
        fractionalChildDepth = float(childDepth / (total_depth + total_depth / 1000))  # purkkafix, childept==totaldepth

        def interpolateSpreadStacked(childIdx, x):
            a = smoothstep(fractionalChildDepth, 1, x)
            s = 1 - spreadStrength
            a = a * (1 - s) + s
            return lerp(spread_positions[childIdx], stacked_positions[childIdx], a)

        for i, childNode in enumerate(childnodes):
            childFraction = childNode['fraction']
            initialSize = childNode['initialSize']

            def doInterpolateSpreadStacked(childIdx, x):
                return stacked_positions[childIdx] if initialSize > 0 else interpolateSpreadStacked(childIdx, x)

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
    def draw_node(node):

        # Segment count. Higher number produces smoother curves.
        sc = 100

        # Find the first segment where the subclone starts to emerge
        first_segment = 0
        for i in range(sc + 1):
            x = i / sc

            if (shapers[str(node['cluster'])](x, 0) - shapers[str(node['cluster'])](x, 1)) != 0:
                # The one where upper and lower edges collide
                first_segment = max(0, i - 1)
                break

        # Start the path
        p = draw.Path(id="clone_" + str(node["cluster"]) + "_" + str(node["parent"]), fill=node["color"],
                      fill_opacity=1.0)
        p.M(first_segment / sc, shapers[str(node['cluster'])](first_segment / sc, 1))

        for i in range(first_segment + 1, sc + 1):
            x = i / sc
            p.L(x, shapers[str(node['cluster'])](x, 1))

        for i in range(sc, first_segment, -1):
            x = i / sc
            p.L(x, shapers[str(node['cluster'])](x, 0))

        g.append(p)

        childnodes = tree.vs.select(parent=node['cluster'])
        for i, childNode in enumerate(childnodes):
            draw_node(childNode)

    draw_node(tree.vs.find(0))

    return g

def addTreeToSvgGroupV1(tree: igraph.Graph, g, rootcluster=0):
    totalDepth = getDepth(tree.vs.find(0))
    # df = tree.get_vertex_dataframe()
    # print(df[['cluster','parent','fraction']])
    # totalDepth = len(df['parent'].unique())
    # graph.get_all_shortest_paths(graph.vs.find(cluster=startcluster)
    print("totalDepth", totalDepth)

    def drawNode(node, shaper, depth=0):
        # print(node)
        if shaper:
            sc = 100  # Segment count. Higher number produces smoother curves.

            firstSegment = 0
            for i in range(sc + 1):
                x = i / sc

                if shaper(x, 0) - shaper(x, 1) != 0:
                    firstSegment = max(0, i - 1)
                    break

            # p = svgwrite.path.Path()
            p = draw.Path(id="clone_" + str(node["cluster"]) + "_" + str(node["parent"]), fill=node["color"],
                          fill_opacity=100.0)
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
        # childnodes = node.successors()
        # print("childnodes:",childnodes)
        spreadPositions = stackChildrenV1(childnodes, node, True)
        stackedPositions = stackChildrenV1(childnodes, node, False)

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

            drawNode(childNode, childShaper, childDepth)

    if rootcluster != 0:
        root = tree.vs.find(cluster=rootcluster)
        pseudoRoot = dict(fraction=float(1.0), parent=0, cluster=rootcluster, initialSize=root['initialSize'],
                          color=root['color'], sample=root['sample'])
    else:
        pseudoRoot = dict(fraction=float(1.0), parent=0, cluster=0, initialSize=1, color='#cccccc', sample="pseudo")
    # pseudoRoot = tree.add_vertex(fraction = float(1.0), parent = 0, cluster = 1, color="#cccccc", sample="pseudo")
    # drawNode(tree.vs.find(parent=0), lambda x, y: y, 0)
    drawNode(pseudoRoot, None, 0)

    return g

def addSampleToSvgGroup(tree: igraph.Graph, phase_graph: igraph.Graph, rootgraph: igraph.Graph, g, sample, translate=[], scale=[], rootcluster=0):
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
            sf = node.successors()
            if sf:
                if sf[0]['frac'] > node['fraction']:
                    node['fraction'] = sf[0]['frac']

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

                for successor in pnode[0].successors() :
                    if successor['cluster'] == node['cluster'] and successor['sample'] != sample:
                        foundfromnextphase = True
                        if node['fraction'] < 0.01:
                            node['fraction'] = 0.02
                    #if successor['parent'] == node['cluster'] and successor['initialSize'] == 0 and node['initialSize'] == 0:
                    #     node['initialSize'] = 1
                    #     print("SUCCS", node['fraction'], successor['fraction'], successor)
                    #     if node['fraction'] < successor['fraction']:
                    #         node['fraction'] = successor['fraction']

                for predecessor in pnode[0].predecessors():
                    if predecessor['cluster'] == node['cluster'] and predecessor['sample'] != sample:
                        foundfromprevphase = True
                if not foundfromprevphase and not foundfromnextphase:
                    node['initialSize'] = 0
                if not foundfromprevphase and foundfromnextphase:
                    node['initialSize'] = 0
                if foundfromprevphase and not foundfromnextphase:
                    node['initialSize'] = 1
                if foundfromprevphase and foundfromnextphase:
                    node['initialSize'] = 1
                # never bellshape if in rootgraph
                if rootgraph.vs.select(cluster=node['cluster']):
                    node['initialSize'] = 1
                schild = tree.vs.select(parent=node['cluster'])
                if schild:
                    if schild[0]['initialSize'] == 0:
                        node['initialSize'] = 1
                #childsampleclone = phase_graph.vs.select(cluster=pnode[0]['cluster'], site=pnode[0]['site'], phase=int(pnode[0]['phase'])+1)
                #print("found",pnode[0],childsampleclone[0])
            #else:
                #node['fraction'] = node['frac']

            height = float(node["fraction"])
            if node['initialSize'] == 0:

                p = draw.Path(id="clone_" +str(sample) + "_" + str(node["cluster"]), fill=node["color"],
                              fill_opacity=1.0)
                csx = 0.2
                csy = yh-lastfrac/2
                cex = 1
                cey = csy - lastfrac/2 #+ lastfrac/10
                cc1x = csx + 0.1
                cc1y = csy + lastfrac/10
                cc2x = cex - 0.1
                cc2y = cey

                p.M(csx, csy)
                p.C(cc1x, cc1y, cc2x, cc2y, cex, cey + lastfrac/10).L(cex, cey + lastfrac).C(cc2x, csy+lastfrac/2, csx-0.1, csy-lastfrac/10, csx, csy)

            else:
                p = draw.Rectangle(firstSegment / sc, yh, 1, height, id="clone_" +str(sample) + "_" + str(node["cluster"]), fill=node['color'], fill_opacity=1.0, translate=translate, scale=scale)

            # yh = yh + height
            g.append(p)
        else:
            shaper = lambda x, y: y  # Make an initial shaper. Just a rectangle, no bell shape

        childnodes = tree.vs.select(parent=node['cluster'])
        # childnodes = node.successors()
        # print("childnodes:",childnodes)
        spreadPositions = stack_children(childnodes, node, True)
        stackedPositions = stack_children(childnodes, node, False)

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


def calculate_sample_position(sample_name, phase_graph, i, height):
    wspace = 200
    mod = i % 2

    #top = height/2-i*200 if mod == 0 else height/2+i*200
    top = i*200
    samplevx = phase_graph.vs.select(sample=sample_name)[0]
    samplenum = int(samplevx['samplenum'])
    phase = int(samplevx['phase'])
    sitenum = int(samplevx['sitenum'])
    left = 500 + (phase-1)*wspace + (samplenum) * wspace
    #top = top+(sitenum+1)*150

    return [left,top]

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
                # endcluster = self.graph.vs.find(i)['cluster']
                # dfendc = self.data.loc[self.data['cluster'] == endcluster]
                # c = 0
                # for index, row in dfendc.iterrows():
                #     print(row)
                #     if row['frac'] > 0:
                #         c += 1
                # if c == 1:
                #     dropouts.add(endcluster)
                gp = self.graph.get_all_simple_paths(0, i, mode='all')
                if len(gp) > 0:
                    allpaths.append(gp[0])
            i += 1

        i = 0
        for index in self.graph.get_adjlist():
            if len(index) == 0 and depth > 2:
                endcluster = self.graph.vs.find(i)['cluster']
                if len(self.data.loc[self.data['cluster'] == endcluster]) < 2:
                    print("sga", index, i, endcluster)
                    dropouts.add(endcluster)
            i += 1

        # TODO: logic for inheriting clone from previous sample, build sample level clonal tree

        #dropouts.add(5)
        #dropouts.add(2)
        root_graph_builder = graph_builder.GraphBuilder(self.data)
        phase_graph = root_graph_builder.build_phase_graph(self.data, dropouts)
        i = 0
        for index in phase_graph.get_adjlist():
            if len(index) < 2 and depth > 2:

                endvs = phase_graph.vs.find(i)
                endcluster = endvs['cluster']
                sameclusterp1 = phase_graph.vs.select(cluster=endcluster, phase=1)
                sameclusterp2 = phase_graph.vs.select(cluster=endcluster, phase=2)
                sameclusterp3 = phase_graph.vs.select(cluster=endcluster, phase=3)

                if len(sameclusterp1) == 0:
                    dropouts.add(endcluster)
                else:
                    if len(phase_graph.vs.select(phase=2)) == 0:
                        if len(sameclusterp1) == 1 and len(sameclusterp3) <= 1:
                            dropouts.add(endcluster)

            i += 1

        # Check if cluster is not present in all phase 1 samples and add it to dropouts
        # phasedf = phase_graph.get_vertex_dataframe().reset_index()
        # phase1 = phasedf.loc[phasedf['phase'] == 1]
        # phasegt1 = phasedf.loc[phasedf['phase'] > 1]
        # los = []
        # pgt1g = phase1.groupby("sample")
        # for gname, g in pgt1g:
        #     print(g['cluster'])
        #     los.extend(g['cluster'].to_list())
        #     #all.add(g['cluster'])
        # vcs = pd.value_counts(los)
        # for vc in vcs.items():
        #     print("VC",vc[0],vc[1])
        #     if vc[1] < len(pgt1g.groups):
        #         dropouts.add(vc[0])
        #
        # cont = phasegt1['cluster'].isin(phase1['cluster'])
        # i = 0
        # for r in cont:
        #     if not r:
        #         dropouts.add(phasegt1.iloc[i]['cluster'])
        #     i += 1
        print("dropouts", dropouts)

        rootgraph = root_graph_builder.build_graph_sep(list(dropouts), 1, True)

        # TODO: cluster the root clones by divergence and split the JellyBell to k clusters
        # root width
        ngroups = len(self.data.groupby("sample").groups) - len(masksample) + 1
        height = ngroups * 250
        width = 2000
        drawing = draw.Drawing(width, height)
        ip.add_axes(drawing)

        # addAxes(d)

        rw = 250
        rh = 300

        transY = 0 #(height / 2) - rh / 2
        # transY=0
        container = draw.Group(id='container', transform="translate(0," + str(transY) + ")")
        drawing.append(container)
        # ImageProcessor.add_axes(self,container)

        rootgroup = draw.Group(id='roog', transform="translate(0," + str((height / 4)) + ") scale(" + str(rw) + "," + str(rh) + ")")
        shapers = tree_to_shapers(rootgraph)

        rootjelly = addTreeToSvgGroupV1(rootgraph, rootgroup)
        container.append(rootjelly)
        tmppng = "./tmp_rootc.png"
        drawing.save_png(tmppng)
        # container.append(composeSimpleJellyBell(self.graph, self.graph.vs.find(cluster=11), self.graph.vs.find(cluster=17),299, 300, 400, 400))

        # edgelist = self.graph.get_edgelist()
        sampleboxes = {}
        tentacles = {}

        # box initial size
        scalex = 100
        scaley = 150
        # transY
        # TODO class object for each element so that its location and dimensions can be determined afterwards
        # print(grouped_samples.groups)

        drawn_clusters = []

        # TODO: group/combine(show just most presentative) the similar samples by using divergence/correlation
        gtype = "p"

        self.data['phase'] = self.data['sample']
        self.data['site'] = self.data['sample']
        for index, row in self.data.iterrows():
            self.data['phase'].at[index] = graph_builder.getPhaseFromSampleName(self.data['sample'].at[index])
        for index, row in self.data.iterrows():
            self.data['site'].at[index] = graph_builder.getSiteFromSampleName(self.data['sample'].at[index])

        preserved_range = [range(-1, -1)]

        grouped_phases = self.data.sort_values(['site'], ascending=False).reset_index().groupby("phase")

        for gname, phase in grouped_phases:
            # Group all elements linked to this sample
            # print("Z", group_name)
            i = 0
            grouped_samples = phase.sort_values(['sample'], ascending=False).reset_index().groupby("sample")
            for sample_name, sample in grouped_samples:

                if sample_name not in masksample:
                    # print("gn", group_name)

                    # print("##"+group_name)
                    # box left pos
                    translate = calculate_sample_position(sample_name, phase_graph, i, height)
                    label = {
                        'text': sample_name,
                        'fontSize': '18',
                        'fill': 'black',
                        'x': translate[0],
                        'y': translate[1] + 140
                    }
                    sampleGroup = draw.Group(id=sample_name)
                    sampleGroup.append(draw.Text(**label, font_size=18))

                    sample_container = draw.Group(id=sample_name,
                                                  transform="translate(" + str(translate[0]) + ", " + str(translate[1]) + ") scale(" + str(
                                                      scalex) + "," + str(scaley) + ")", x=translate[0], y=translate[1])

                    gr = sample.sort_values(['dfs.order'], ascending=True)
                    sample_graph_builder = graph_builder.GraphBuilder(gr)
                    sample_graph = sample_graph_builder.build_graph_sep_sample(list(dropouts))

                    samplebox = addSampleToSvgGroup(sample_graph, phase_graph, rootgraph, sample_container, sample_name, translate, [scalex,scaley])
                    #samplejelly = addSampleToSvgGroup(sample_graph, treeToShapers(sample_graph, 0), sample_container)
                    container.append(samplebox)
                    sampleboxes[sample_name] = samplebox
                    container.append(sampleGroup)
                    i=i+1


        drawing.save_png(tmppng)  # TODO: do in-memory
        img = Image.open(tmppng)  # Specify image path
        img_processor = ip.ImageProcessor(img)
        # TODO: Move tentacle drawing after samplegroup drawung, because now all the boxes may not be present when connecting AND use the get_el_pos_by_id()

        # Draw tentacles
        drawn_tentacles = []
        samplegroups = self.data.groupby('sample')
        for group_name, group in samplegroups:
            if group_name not in masksample:
                sampleboxpos = get_el_pos_of_group(sampleboxes[group_name])
                prevboxpos = None
                print("sampleboxpos", group_name, sampleboxpos)

                samplevertexes = phase_graph.vs.select(sample=group_name)
                conn_prevphase = False
                for svertex in samplevertexes:
                    preds = svertex.predecessors()
                    for p in preds:
                        if svertex['phase'] > 1 and p['site'] == svertex['site'] and not p['sample'] == svertex['sample'] and p['phase'] <= svertex['phase']:
                            print("HERE", p['sample'], svertex['sample'], p['site'], p['cluster'])
                            conn_prevphase = True
                            prevboxpos = get_el_pos_of_group(sampleboxes[p['sample']])

                for index, row in group.iterrows():
                    # if top < 0:
                    cluster = row['cluster']

                    if cluster > -1:
                        # print(cluster)
                        # nextv = self.graph.vs.find(parent=cluster)
                        if True: #int(cluster) not in dropouts:

                            # Draw tentacle paths
                            if conn_prevphase:
                                left = int(sampleboxpos[0])
                                rx = int(prevboxpos[0])+scalex-2
                                offsy1 = scaley-20

                                sy1 = (int(prevboxpos[1]) + int(transY) + offsy1) if (int(prevboxpos[1]) + int(
                                    transY) + offsy1) < height else height - offsy1
                                sy2 = (int(prevboxpos[1]) + int(transY) + int(scaley * 2.5)) if (int(prevboxpos[1]) + int(
                                    transY) + int(scaley * 2.5)) < height else height
                                y1 = (int(sampleboxpos[1]) + int(transY) + scaley) if (int(sampleboxpos[1]) + int(
                                    transY) + scaley) < height else height - scaley
                                y2 = (int(sampleboxpos[1]) + int(transY) + int(scaley*2.5)) if (int(sampleboxpos[1]) + int(
                                    transY) + int(scaley * 2.5)) < height else height


                                print("y1y2", cluster, group_name, sy1, sy2)
                                ystartrange = img_processor.extract_point_by_cluster_color(rx, rx + 1, sy1, sy2,
                                                                                             row['color'])

                                starty = ystartrange[0] + (ystartrange[1] - ystartrange[0]) / 2 - transY  # (-1*transY)-ypoints[1]+(ypoints[1]-ypoints[0])/2
                                p = draw.Path(id="tnt" + str(cluster) + "_" + str(group_name), stroke_width=2,
                                              stroke=row['color'], fill=None, fill_opacity=0.0)
                                p.M(rx, float(starty))  # Start path at point
                                yendrange = img_processor.extract_point_by_cluster_color(left + 2, left + 3,
                                                                                         y1, y2,
                                                                                         row['color'])
                                print("HERE1", cluster, group_name, rx, starty, ystartrange, yendrange)
                                if yendrange[0] != 0 and yendrange[1] != 0 and starty != 0 and [group_name,int(cluster)] not in drawn_tentacles:
                                    preserved_range.append(range(yendrange[0], yendrange[1]))
                                    endy = yendrange[0] + (yendrange[1] - yendrange[0]) / 2 - transY
                                    print("HERE2", cluster, group_name, rx, starty, left, endy)
                                    bz2ndy = endy
                                    bz2ndx = (left - int(samplevertexes[0]['samplenum']) * 25)
                                    p.C(rx + 25, float(starty) + 10, bz2ndx, bz2ndy, left + 1, endy)
                                    if [group_name, cluster] not in drawn_tentacles:
                                        drawn_tentacles.append([group_name, int(cluster)])

                            else:
                            #draw from root bell
                                print("fromroot", group_name)
                                rx = rw-2
                                left = 500
                                y1 = 0
                                y2 = height
                                if sampleboxpos:
                                    offy1 = 150
                                    offy2 = 310
                                    left = int(sampleboxpos[0])
                                    y1 = (int(sampleboxpos[1])+int(transY)+offy1) if (int(sampleboxpos[1])+int(transY)+offy1) < height else height-offy1
                                    y2 = (int(sampleboxpos[1]) + int(transY) + offy2) if (int(sampleboxpos[1]) + int(transY) + offy2) < height else height
                                    print("Rooty1y2",str(transY), group_name, y1, y2)

                                ystartrange = img_processor.extract_point_by_cluster_color(rx - 1, rx, 0,
                                                                                             height,
                                                                                             row['color'])
                                starty = ystartrange[0] + (ystartrange[1] - ystartrange[
                                    0]) / 2 - transY  # (-1*transY)-ypoints[1]+(ypoints[1]-ypoints[0])/2
                                p = draw.Path(id="tnt" + str(cluster) + "_" + str(group_name), stroke_width=2,
                                              stroke=row['color'], fill=None, fill_opacity=0.0)
                                p.M(rx, float(starty))  # Start path at point
                                yendrange = img_processor.extract_point_by_cluster_color(left + 1, left + 2, y1, y2,
                                                                                           row['color'])

                                print("HERE3", cluster, group_name, rx, starty, left, ystartrange, yendrange)

                                if yendrange[0] != 0 and yendrange[1] != 0 and starty != 0 and [group_name,int(cluster)] not in drawn_tentacles:
                                    preserved_range.append(range(yendrange[0], yendrange[1]))
                                    endy = yendrange[0] + (yendrange[1] - yendrange[0]) / 2 - transY
                                    print("HERE4", cluster, group_name, rx, starty, left, endy)

                                    bz2ndy = endy
                                    if gtype == "p":
                                        bz2ndx = (left - left / 4)
                                    if gtype == "i":
                                        bz2ndx = (left - left / 1.5)
                                    if gtype == "r":
                                        bz2ndx = (left - left / 1.5)
                                    p.C(rx + left / 4, float(starty) + 10, bz2ndx, bz2ndy, left + 1, endy)
                                    if [group_name, cluster] not in drawn_tentacles:
                                        drawn_tentacles.append([group_name, int(cluster)])

                            container.append(p)

                            if cluster not in drawn_clusters:
                                drawn_clusters.append(int(cluster))

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
        ip.add_axes(container)
        return drawing
