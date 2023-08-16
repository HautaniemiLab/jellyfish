import drawsvg as draw
import igraph
import pandas as pd
from PIL import Image
import graph_builder
import image_processor as ip
import sample_analyzer


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


def fancystep(edge0, edge1, x, tipShape):
    span = edge1 - edge0
    step = lambda x: smootherstep(edge0 - span * (1 / (1 - tipShape) - 1), edge1, x)
    atZero = step(edge0)
    return float(max(0, step(x) - atZero) / (1 - atZero))


def stackChildren(nodes, node, spread=False):
    # print(nodes)
    # fractions = [float(n.get('fraction')) / float(node.get('fraction')) for n in node.get('children')]
    fractions = []
    for n in nodes:
        if node['fraction'] == 0.0:
            node['fraction'] = 1.0
        fraction = float(n['fraction']) / float(node['fraction'])
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


def lerp(a, b, x):
    return float((1 - x) * a + x * b)


tipShape = 0.1
spreadStrength = 0.5


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


def addTreeToSvgGroup(tree: igraph.Graph, g, tipShape, spreadStrength, rootcluster=0):
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
        depth = len(self.data['parent'].unique())
        for index in self.graph.get_adjlist():
            if index == [] and depth > 2:
                endvertices.add(i)
                endcluster = self.graph.vs.find(i)['cluster']
                dropouts.add(endcluster)
                gp = self.graph.get_all_simple_paths(0, i, mode='all')
                if len(gp) > 0:
                    allpaths.append(gp[0])
            i += 1
        print("dropouts", dropouts)
        root_graph_builder = graph_builder.GraphBuilder(self.data)
        rootgraph = root_graph_builder.build_graph_sep(list(dropouts), 0, True)
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
        rootjelly = addTreeToSvgGroup(rootgraph, rootgroup, tipShape, spreadStrength)
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
        left = 500
        # print(grouped_samples.groups)

        drawn_clusters = []

        # TODO: group/combine(show just most presentative) the similar samples by using divergence/correlation
        gtype = "p"
        samplenum = 0

        self.data['phase'] = self.data['sample'].str[0:2]
        phases = set(self.data['phase'].unique().tolist())
        print(phases)
        preserved_range = [range(-1, -1)]
        left = 500
        sorted_groups = grouped_samples.groups.keys()
        sorted_groups = reversed(sorted(sorted_groups))
        for key in sorted_groups:
            # Group all elements linked to this sample
            # print("Z", group_name)

            group = grouped_samples.get_group(key)
            group_name = key
            if group_name not in masksample:
                # print("gn", group_name)

                # print("##"+group_name)
                # box left pos
                samplenum = str(group_name)[1]
                if group_name.startswith("p"):
                    left = 500
                    if samplenum.isnumeric():
                        if group_name[0] + str(int(samplenum) - 1) in phases or self.data['phase'].str.match(
                                "^p{1}[A-Z]$") is not None:
                            if int(samplenum) > 1:
                                left = left + (int(samplenum) - 1) * 200
                                top = top - 210
                    gtype = "p"
                if group_name.startswith("i"):
                    if "p" not in list(self.data['phase'].str[0]):
                        left = 500
                    else:
                        left = 700
                    if samplenum.isnumeric():
                        if group_name[0] + str(int(samplenum) - 1) in phases or self.data['phase'].str.match(
                                "^i{1}[A-Z]$") is not None:
                            if int(samplenum) > 1:
                                left = left + (int(samplenum) - 1) * 200
                                top = top - 210

                    gtype = "i"
                if group_name.startswith("r"):

                    if "i" not in list(self.data['phase'].str[0]) or "p" not in self.data['phase'].str[0]:
                        left = 700
                    else:
                        left = 900
                    if samplenum.isnumeric():
                        if group_name[0] + str(int(samplenum) - 1) or self.data['phase'].str.match(
                                "^r{1}[A-Z]$") is not None:
                            if int(samplenum) > 1:
                                left = left + (int(samplenum) - 1) * 200
                                top = top - 210
                    gtype = "r"

                top += 100

                label = {
                    'text': group_name,
                    'fontSize': '18',
                    'fill': 'black',
                    'x': left,
                    'y': top - 10
                }
                sampleGroup = draw.Group(id=group_name)
                sampleGroup.append(draw.Text(**label, font_size=18))
                sample_container = draw.Group(id=group_name,
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

                samplejelly = addTreeToSvgGroup(sample_graph, sample_container, tipShape, spreadStrength, 0)
                container.append(samplejelly)
                drawing.save_png(tmppng)  # TODO: do in-memory
                img = Image.open(tmppng)  # Specify image path
                image_processor = ip.ImageProcessor(img)
                for index, row in gr.iterrows():

                    # if top < 0:
                    cluster = row['cluster']

                    vertex = self.graph.vs.find(cluster=row['cluster'])
                    frac = row['frac']
                    sbheight = float(y) * float(frac)

                    if cluster > -1:

                        # print(cluster)
                        if not (vertex.index in endvertices):
                            # nextv = self.graph.vs.find(parent=cluster)
                            inc_bell = False
                            outedges = vertex.out_edges()
                            for edge in outedges:
                                target = edge.target
                                tv = self.graph.vs.find(target)

                                # path_to_end = self.graph.get_all_simple_paths(edge.target, mode='in')
                                # self.graph.es.find(target)

                                if target in endvertices and tv['cluster'] in gr['cluster'].tolist():
                                    # if tv['cluster'] in gr['cluster'].tolist():

                                    # TODO: if multiple jbs inside cluster, combine to new jellybell starting from parent (check H032)
                                    targetdata = self.data.loc[
                                        (self.data['cluster'] == tv['cluster']) & (self.data['sample'] == group_name)]
                                    targetfrac = targetdata['frac'].values[0]
                                    # print(tv['cluster'],parentfrac.values[0])
                                    if targetfrac > frac_threshold:

                                        if targetfrac >= frac:
                                            sbheight = targetfrac * y
                                        # jb = JellyBellComposer.compose_simple_jelly_bell(data, graph, sbheight, x, left, top, vertex.index, tv.index)
                                        # Draw new jellybelly inside clone
                                        inc_bell = True
                                        # boxjbs.append(jb)
                                        if tv['cluster'] not in drawn_clusters:
                                            drawn_clusters.append(int(tv['cluster']))
                                        # Check with H023, cluster 6 inside 2, if this indentation increased -> fixed partly

                            if frac > frac_threshold:
                                cluster = row['cluster']

                                if samplenum.isnumeric() and int(samplenum) > 1:
                                    rx = left - 101
                                    ystartrange = image_processor.extract_point_by_cluster_color(rx, rx + 1, 0, height,
                                                                                                 row['color'])

                                    starty = ystartrange[0] + (ystartrange[1] - ystartrange[
                                        0]) / 2 - transY  # (-1*transY)-ypoints[1]+(ypoints[1]-ypoints[0])/2
                                    p = draw.Path(id="tnt" + str(cluster) + "_" + str(group_name), stroke_width=2,
                                                  stroke=row['color'], fill=None, fill_opacity=0.0)
                                    p.M(rx, float(starty))  # Start path at point
                                    yendrange = image_processor.extract_point_by_cluster_color(left + 2, left + 3,
                                                                                               int(0), int(height),
                                                                                               row['color'],
                                                                                               preserved_range)
                                    if yendrange[0] != 0 and yendrange[1] != 0:
                                        preserved_range.append(range(yendrange[0], yendrange[1]))
                                        endy = yendrange[0] + (yendrange[1] - yendrange[0]) / 2 - transY
                                        if inc_bell:
                                            endy = yendrange[0] - transY
                                        bz2ndy = endy
                                        bz2ndx = (left - 25)
                                        p.C(rx + 25, float(starty), bz2ndx, bz2ndy, left, endy)
                                else:
                                    rx = rw
                                    ystartrange = image_processor.extract_point_by_cluster_color(rx - 1, rx, 0, height,
                                                                                                 row['color'])
                                    starty = ystartrange[0] + (ystartrange[1] - ystartrange[
                                        0]) / 2 - transY  # (-1*transY)-ypoints[1]+(ypoints[1]-ypoints[0])/2
                                    p = draw.Path(id="tnt" + str(cluster) + "_" + str(group_name), stroke_width=2,
                                                  stroke=row['color'], fill=None, fill_opacity=0.0)
                                    p.M(rx, float(starty))  # Start path at point
                                    yendrange = image_processor.extract_point_by_cluster_color(left + 2, left + 3,
                                                                                               int(0), int(height),
                                                                                               row['color'],
                                                                                               preserved_range)
                                    if yendrange[0] != 0 and yendrange[1] != 0:

                                        preserved_range.append(range(yendrange[0], yendrange[1]))
                                        endy = yendrange[0] + (yendrange[1] - yendrange[0]) / 2 - transY
                                        if inc_bell:
                                            endy = yendrange[0] - transY
                                        bz2ndy = endy
                                        if gtype == "p":
                                            bz2ndx = (left - left / 4)
                                        if gtype == "i":
                                            bz2ndx = (left - left / 3)
                                        if gtype == "r":
                                            bz2ndx = (left - left / 2)
                                        p.C(rx + left / 4, float(starty), bz2ndx, bz2ndy, left, endy)

                                sampleGroup.append(p)

                                if cluster not in drawn_clusters:
                                    drawn_clusters.append(int(cluster))

                                    # print("HERE11",group_name, cluster)
                                    # svggr.append(draw.Text(str(cluster), 12, path=p, text_anchor='end', valign='middle'))

                            else:
                                # if row['parent'] > 0:
                                # print(row['parent'],self.data.loc[self.data['cluster'] == row['parent']]['color'].values[0])
                                cluster = row['parent']
                                if cluster == -1:
                                    cluster = 1
                                parent = self.data.loc[
                                    (self.data['cluster'] == cluster) & (self.data['sample'] == group_name)]
                                # print(group_name, row['cluster'], parent)
                                # frac = parent['frac'].values[0]

                                if int(cluster) not in dropouts:  # TODO: this sbheight filter is purkkafix, use parent fraction or better is to change logic so that same cluster is processed just once

                                    # Draw tentacle paths
                                    if samplenum.isnumeric() and int(samplenum) > 1:
                                        rx = left - 101
                                        ystartrange = image_processor.extract_point_by_cluster_color(rx, rx + 1, 0,
                                                                                                     height,
                                                                                                     row['color'])

                                        starty = ystartrange[0] + (ystartrange[1] - ystartrange[
                                            0]) / 2 - transY  # (-1*transY)-ypoints[1]+(ypoints[1]-ypoints[0])/2
                                        p = draw.Path(id="tnt" + str(cluster) + "_" + str(group_name), stroke_width=2,
                                                      stroke=row['color'], fill=None, fill_opacity=0.0)
                                        p.M(rx, float(starty))  # Start path at point
                                        yendrange = image_processor.extract_point_by_cluster_color(left + 10, left + 11,
                                                                                                   int(0), int(height),
                                                                                                   row['color'],
                                                                                                   preserved_range)
                                        if yendrange[0] != 0 and yendrange[1] != 0:
                                            preserved_range.append(range(yendrange[0], yendrange[1]))
                                            endy = yendrange[0] + (yendrange[1] - yendrange[0]) / 2 - transY
                                            if inc_bell:
                                                endy = yendrange[0] - transY
                                            bz2ndy = endy
                                            bz2ndx = (left - 25)
                                            p.C(rx + 25, float(starty) + 10, bz2ndx, bz2ndy, left, endy)
                                    else:
                                        rx = rw
                                        ystartrange = image_processor.extract_point_by_cluster_color(rx - 1, rx, 0,
                                                                                                     height,
                                                                                                     row['color'])
                                        starty = ystartrange[0] + (ystartrange[1] - ystartrange[
                                            0]) / 2 - transY  # (-1*transY)-ypoints[1]+(ypoints[1]-ypoints[0])/2
                                        p = draw.Path(id="tnt" + str(cluster) + "_" + str(group_name), stroke_width=2,
                                                      stroke=row['color'], fill=None, fill_opacity=0.0)
                                        p.M(rx, float(starty))  # Start path at point
                                        yendrange = image_processor.extract_point_by_cluster_color(left + 1, left + 2,
                                                                                                   int(0), int(height),
                                                                                                   row['color'],
                                                                                                   preserved_range)
                                        if yendrange[0] != 0 and yendrange[1] != 0:

                                            preserved_range.append(range(yendrange[0], yendrange[1]))
                                            endy = yendrange[0] + (yendrange[1] - yendrange[0]) / 2 - transY
                                            if inc_bell:
                                                endy = yendrange[0] - transY
                                            bz2ndy = endy
                                            if gtype == "p":
                                                bz2ndx = (left - left / 4)
                                            if gtype == "i":
                                                bz2ndx = (left - left / 3)
                                            if gtype == "r":
                                                bz2ndx = (left - left / 2)
                                            p.C(rx + left / 4, float(starty) + 10, bz2ndx, bz2ndy, left, endy)

                                    sampleGroup.append(p)

                                    if cluster not in drawn_clusters:
                                        drawn_clusters.append(int(cluster))

                            top = top + sbheight
                            # top = top+y/ns

                            # toff = rootarcs[i][0].args['d'].split(',')[2]
                            # if top < 0:
                    for jb in boxjbs:
                        sampleGroup.append(jb)

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
