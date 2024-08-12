import drawsvg as draw
import igraph
import pandas as pd
from PIL import Image
import graph_builder
import image_processor as ip
import sample_analyzer


tipShape = 0.1
spreadStrength = 0.5

def get_el_pos_by_id(svggroup: draw.Group, context, id):
    for el in svggroup.all_children(context):
        groupscaley = svggroup.args['scaley']
        if isinstance(el, draw.elements.Path) == True:
            if str(el.id) == id:
                args = el.args['d'].split(' ')
                M = args[0].split(',')
                C = args[1].split(',')
                Mx = float(M[0][1:])
                My = float(M[1])
                t = el.args['translate']
                s = el.args['scale']

                #print("PATH",id, Mx, My, s, t)
                #return [Mx, My, s, t]
                #return [Mx, My*float(s[1]),  Mx*float(s[0])+t[0], My*float(s[1])+t[1]]

                tpy = el.args['tpy']
                #print("TEEPEE",tpy, groupscaley)
                #return [Mx + float(t[0]), tpy * float(groupscaley)+float(t[1]/2), float(s[0]), (tpy * float(groupscaley)+float(t[1])/2)]
                return [Mx + float(t[0]), tpy*float(groupscaley)+float(t[1])]

                #starty = float(startpos[1]) + float(startpos[3]) / 2
                #endy = float(endpos[1]) + float(endpos[3]) / 2 - transY
        if isinstance(el, draw.elements.Rectangle) == True:
            if str(el.id) == id:
                #print(el.id, id)
                #print(el.args)
                x = float(el.args['x'])
                y = float(el.args['y'])
                w = float(el.args['width'])
                h = float(el.args['height'])
                t = el.args['translate']
                s = el.args['scale']
                print("RECTA",id, x, y, w, h, s, t)
                #return [x, y, s, t]
                #return [x*float(s[0])+float(t[0]), y*float(groupscaley)+float(t[1]), w*float(s[0]), h*float(groupscaley)]
                return [x * float(s[0]) + float(t[0]), y * float(groupscaley) + (h * float(groupscaley))/2 + float(t[1])]

def get_el_pos_of_group(el: draw.Group):
    x = float(el.args['x'])
    y = float(el.args['y'])
    scaley = el.args['scaley']
    return [x, y, scaley]

def scale_group_height(el: draw.Group, currheight, prefheight, scalex):
    x = float(el.args['x'])
    y = float(el.args['y'])
    scaley = prefheight/currheight
    el.args['transform'] = "translate("+str(x)+"," + str(y) + ") scale(" + str(scalex) + "," + str(scaley) + ")"
    el.args['scaley'] = scaley
    return el

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
    #fractions = [float(n['proportion']) / float(node['proportion']) for n in node['proportion']]
    fractions = []
    for n in childnodes:
        fraction = float(n['proportion'])
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

def stack_children2(childnodes, node, spread=False):
    # print(nodes)
    #fractions = [float(n['proportion']) / float(node['proportion']) for n in node['<proportion>']]
    fractions = []

    fractions.append(float(node['proportion']))

    return fractions


def stackTree(tree: igraph.Graph, shapers: dict, edge=1):
    stackedNodes = dict()

    def process(node):

        top = shapers[str(node['subclone'])](edge, 0)
        bottom = shapers[str(node['subclone'])](edge, 1)
        childnodes = tree.vs.select(parent=node['subclone'])
        for child in childnodes:
            bottom = min(bottom, process(child))
        stackedNodes[str(node['subclone'])] = [top, bottom]

        return top if bottom - top > 0 else 1

    process(tree.vs.find(0))
    return stackedNodes


def lerp(a, b, x):
    return float((1 - x) * a + x * b)

def get_all_children(g, rootsubclone):
    # Get the root vertex
    root = g.vs.find(subclone=rootsubclone)
    # Recursively normalize fractions
    childrenids = set()

    def get_children(vertex):
        children = vertex.successors()
        #print(children)
        for child in children:
            childrenids.add(child.index)
            get_children(child)

    get_children(root)
    return list(childrenids)


def tree_to_shapers(tree: igraph.Graph, rootsubclone=1):
    total_depth = getDepth(tree.vs.find(subclone=rootsubclone))

    shapers = dict()

    def process(node, shaper, depth=0):

        if shaper is None:
            shaper = lambda x, y: y  # Make an initial shaper. Just a rectangle, no bell shape

        shapers[str(node['subclone'])] = shaper

        childnodes = tree.vs.select(parent=node['subclone'])
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
            childFraction = childNode['proportion']
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

    if rootsubclone != 0:
        root = tree.vs.find(subclone=rootsubclone)
        pseudoRoot = root
        #pseudoRoot['initialSize'] = 1
        #pseudoRoot['proportion'] = float(1.0)
        #pseudoRoot['parent'] = 0
    else:
        ig = igraph.Graph(directed=True)
        pseudoRoot = ig.add_vertex()
        pseudoRoot['proportion'] = float(1.0)
        pseudoRoot['parent'] = 0
        pseudoRoot['subclone'] = 0
        pseudoRoot['initialSize'] = 1
        pseudoRoot['color'] = '#cccccc'
        pseudoRoot['sample'] = "pseudo"

    process(pseudoRoot, None, 0)

    return shapers

def addTreeToSvgGroupV1(tree: igraph.Graph, g, stacked_tree, translate=[], scale=[], rootsubclone=1, inferred = True):
    totalDepth = getDepth(tree.vs.find(0))

    def drawNode(node, shaper, depth=0):
        print(node['sample'],node['subclone'],node['proportion'], node['initialSize'])
        # print(node)
        p = None
        if shaper:
            sc = 100  # Segment count. Higher number produces smoother curves.

            firstSegment = 0
            for i in range(sc + 1):
                x = i / sc

                if shaper(x, 0) - shaper(x, 1) != 0:
                    firstSegment = max(0, i - 1)
                    break

            # p = svgwrite.path.Path()
            id_prefix = "clone_"+str(node['sample'])+"_"
            if node['site'] == 'inferred':
                id_prefix = "clone_root_"
            p = draw.Path(id=id_prefix + str(node['subclone']), fill=node["color"],
                          fill_opacity=100.0, translate=translate, scale=scale)
            # TODO: store right edge middle y point

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

        childnodes = tree.vs.select(parent=node['subclone'])

        spreadPositions = stack_children(childnodes, node, False if inferred else False)
        stackedPositions = stack_children(childnodes, node, False if inferred else False)

        if p:
            #st = stackTree(tree, shaper, 1)
            p.args['tp'] = 0.0
            l = p.args['d'].split('L')[1:]
            df = pd.DataFrame(l)
            df[['x', 'y']] = df[0].str.split(',', expand=True)

            maxy = float(df.max()['y'])
            miny = float(df.min()['y'])
            attach_pointy = (miny+maxy)/2
            if len(stackedPositions) > 0:
                stackpos = stackedPositions[len(stackedPositions)-1]
                attach_pointy = miny + stackpos/2

            p.args['tpy'] = float(attach_pointy) # (float(df.max()['y'])-float(df.min()['y']))/4

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
            childFraction = childNode['proportion']
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

    #total_fraction = sum(tree.vs.select(fraction_gt=0.0)['proportion'])
    if rootsubclone != 1 or inferred == False:
        root = tree.vs.find(0)
        pseudoRoot = dict(proportion=root['proportion'], parent=root['parent'], subclone=root['subclone'], initialSize=1, color=root['color'], sample=root['sample'], site=root['site'])
        drawNode(pseudoRoot, (lambda x, y: y), 0)
    else:
        pseudoRoot = dict(proportion=float(1.0), parent=0, subclone=0, initialSize=1, color='#cccccc', sample="pseudo", site="inferred")
        drawNode(pseudoRoot, None, 0)

    return scale_group_height(g, 1.0, scale[1], scale[0])

def calculate_sample_position(sample_name, phase_graph, i, totalnum, maxsamplesinphases, height):
    wspace = 200

    samplevx = phase_graph.vs.select(sample=sample_name)[0]
    #samplenum = int(samplevx['samplenum'])
    rank = int(samplevx['rank'])
    #sitenum = int(samplevx['sitenum'])

    # top = height/2-i*200 if mod == 0 else height/2+i*200
    mod = i % 2  # +samplenum
    mi = 1
    if mod == 0:
        mi = -1
    middle = height / 2
    hspace = height / (maxsamplesinphases)
    top = middle + (mi * (i + 1) * hspace) / 2
    if totalnum == 1:
        top = middle + mod * (hspace / 2)

    # TODO: check OC034, H178, H112, H092
    left = 500 + rank * wspace
    # top = top+(sitenum+1)*150
    # print("calculate_sample_position", sample_name, i, middle, hspace, height, left, top)
    return [left, top]


def has_connection_to_prev_phase(phase_graph, sample_name):
    samplevertexes = phase_graph.vs.select(sample=sample_name)
    conn_prevphase = False
    for svertex in samplevertexes:
        preds = svertex.predecessors()
        for p in preds:
            if svertex['rank'] > 0 and p['site'] == svertex['site'] and not p['sample'] == svertex[
                'sample'] and p['rank'] <= svertex['rank']:
                return p['sample']
    return "root"


def draw_tentacle(vertex, child, sampleboxes, drawing, transY, scalex, rw):
    group_name = child['sample']
    conn_prevphase_sample = vertex['sample']
    #sampleboxpos = get_el_pos_of_group(sampleboxes[child['sample']])
    cluster = child['subclone']

    #left = int(sampleboxpos[0])

    endpos = get_el_pos_by_id(sampleboxes[group_name], drawing, "clone_" + str(group_name) + "_" + str(child['subclone']))
    startpos = get_el_pos_by_id(sampleboxes[conn_prevphase_sample], drawing,
                                "clone_" + str(conn_prevphase_sample) + "_" + str(child['subclone']))
    # print(conn_prevphase_sample,cluster,startpos)
    if startpos == None or endpos == None:
        return None

    starty = float(startpos[1])
    endy = float(endpos[1]) - transY

    startx = startpos[0] + scalex
    if conn_prevphase_sample == "root":
        startx = startpos[0] + rw
    endx = endpos[0]

    p = draw.Path(id="tnt" + str(cluster) + "_" + conn_prevphase_sample + "_" + str(group_name), stroke_width=2,
                  stroke=child['color'], fill=None, fill_opacity=0.0)
    p.M(startx, float(starty))  # Start path at point

    #squeez = 20
    #if i <= (len(group) / 2) - 1:
    #    squeez = -1 * squeez
    bz2ndy = endy #+ squeez
    length = endpos[0] - startx
    bz1x = startx + length / 4
    bz2ndx = endpos[0] - length / 4
    p.C(bz1x, float(starty) + 10, bz2ndx, bz2ndy, endx, endy)

    return p


class Drawer:
    def __init__(self, samples: pd.DataFrame, ranks: pd.DataFrame, phylogeny: pd.DataFrame, composition: pd.DataFrame, min_fraction, min_correlation, cfds=None):
        self.samples: pd.DataFrame = samples
        self.ranks: pd.DataFrame = ranks
        self.phylogeny: pd.DataFrame = phylogeny
        self.composition: pd.DataFrame = composition
        self.min_fraction = min_fraction
        self.min_correlation = min_correlation
        #self.cfds = cfds

    def draw(self, scx, scy, patient):

        frac_threshold = self.min_fraction
        corr_treshold = self.min_correlation

        uniqsc = self.composition['subclone'].unique()
        uniqsamples = self.composition['sample'].unique()
        for sc in self.phylogeny['subclone']:
            if sc not in uniqsc:
                for sample in uniqsamples:
                    self.composition._append({'sample':sample, 'subclone':sc, 'proportion':0.00}, ignore_index = True)
        comp_and_phylogeny = self.composition.join(self.phylogeny.set_index('subclone'), on='subclone')
        samples_and_ranks = self.samples.join(self.ranks.set_index('timepoint'), on='timepoint')
        joinedf = comp_and_phylogeny.join(samples_and_ranks.set_index('sample'), on='sample')
        print(joinedf)
        patient_cfds = sample_analyzer.calc_sample_clonal_freqs(self.composition)
        print('patient_cfds',patient_cfds)
        corr_matrix = sample_analyzer.calc_corr_matrix(patient_cfds, patient, True)
        #print(corr_matrix)

        # Find maskable clusters
        dropouts = set()
        masksample = set()
        maskbythreshold = []

        for sample, corrs in corr_matrix.iterrows():
            similar = corrs.loc[corrs.index != sample].loc[corrs > corr_treshold]
            # masksample.add(similar)
            for name in similar.index:
                masksample.add(name)

        print("masked", masksample)

        for index, row in joinedf.iterrows():
            if row['proportion'] < frac_threshold:
                maskbythreshold.append([row['sample'], row['subclone'], row['rank']])

        subclone_cnts = joinedf.groupby('subclone')
        for ind,grp in subclone_cnts:
            print("subclc",ind, len(grp))
            if len(grp) == 1:
                if ind != 1:
                    dropouts.add(ind)

        print("dropouts", dropouts)
        # Exclude root from dropouts
        if 1 in dropouts:
            dropouts.remove(1)

        root_graph_builder = graph_builder.GraphBuilder(joinedf.sort_values("proportion", ascending=False))
        totalgraph = root_graph_builder.build_total_graph(patient, dropouts, frac_threshold, 1, True)

        # Calculate dimensions by max number of samples in phases
        maxsamplesinphase = 0
        pgs = totalgraph.get_vertex_dataframe().reset_index().groupby(['rank'])
        for ind, gmr in pgs:
            if len(gmr['sample'].unique()) > maxsamplesinphase:
                maxsamplesinphase = len(gmr['sample'].unique())

        hmargin = maxsamplesinphase * 150
        height = maxsamplesinphase * 250 + hmargin
        width = 2000
        drawing = draw.Drawing(width, height)
        # ip.add_axes(drawing)
        # addAxes(d)

        rw = 250
        rh = 300

        # box initial size
        scalex = 100
        scaley = 150
        # transY

        transY = 0 #(height / 2) - rh / 2
        rootY = height/4
        # transY=0
        container = draw.Group(id='container', transform="translate(0," + str(rootY) + ")")
        drawing.append(container)

        rootgroup = draw.Group(id='root',
                               transform="translate(0," + str(rootY) + ") scale(" + str(rw) + "," + str(rh) + ")", x=0,
                               y=str(rootY), scaley=str(rh))

        rootnodes = totalgraph.vs.select(site="inferred")
        rootgraph = totalgraph.subgraph(rootnodes)
        shapers = tree_to_shapers(rootgraph, 1)
        stacked_tree = stackTree(rootgraph, shapers, 1)
        print("stacked_tree", stacked_tree)

        rootjelly = addTreeToSvgGroupV1(rootgraph, rootgroup, stacked_tree,[0, rootY], [rw, rh], 1)
        container.append(rootjelly)
        # edgelist = self.graph.get_edgelist()
        sampleboxes = {}
        sampleboxes["root"] = rootjelly
        container.append(rootjelly)

        grouped_phases = joinedf.reset_index().groupby(["rank"])
        # Iterate phase by phase
        for gname, phase in grouped_phases:

            total_samples = len(phase['sample'].unique())
            grouped_samples = phase.sort_values(['sample'], ascending=False).reset_index().groupby("sample")
            i = 0
            for sample_name, sample in grouped_samples:
                if sample_name not in masksample:

                    translate = calculate_sample_position(sample_name, totalgraph, i, total_samples,
                                                          maxsamplesinphase, height - hmargin)

                    sample_container = draw.Group(id=sample_name,
                                                  transform="translate(" + str(translate[0]) + ", " + str(translate[1]) + ") scale(" + str(
                                                      scalex) + "," + str(scaley) + ")", x=translate[0], y=translate[1], scaley=str(scaley))


                    samplenodes = totalgraph.vs.select(sample=sample_name)
                    samplegraph = totalgraph.subgraph(samplenodes)
                    igraph.plot(samplegraph, "./total_graph_un_" + sample_name + ".pdf", centroid=(800, -800), bbox=(1600, 1600),
                                layout="sugiyama")

                    samplebox = addTreeToSvgGroupV1(samplegraph, sample_container, stacked_tree, translate, [scalex, scaley], 0, False)
                    #samplebox = addSampleToSvgGroup(samplegraph, sample_container, sample_name, translate, [scalex, scaley], 0)

                    container.append(samplebox)
                    sampleboxes[sample_name] = samplebox
                    label = {
                        'text': sample_name,
                        'fontSize': '18',
                        'fill': 'black',
                        'x': translate[0],
                        'y': get_el_pos_of_group(sampleboxes[sample_name])[1] - 10
                    }
                    sampleGroup = draw.Group(id=sample_name)
                    sampleGroup.append(draw.Text(**label, font_size=18))
                    container.append(sampleGroup)
                    i=i+1


        # Draw tentacles
        drawn_tentacles = []

        def recursive_walk(vertex):
            children = vertex.successors()
            for child in children:
                if child['site'] != 'inferred':
                    t = draw_tentacle(vertex, child, sampleboxes, drawing, transY, scalex, rw)
                    if t and [child['sample'], int(child['subclone'])] not in drawn_tentacles:
                        container.append(t)
                        if [child['sample'], child['subclone']] not in drawn_tentacles:
                            drawn_tentacles.append([child['sample'], int(child['subclone'])])
                recursive_walk(child)

        rootvertex = totalgraph.vs.find(0)
        recursive_walk(rootvertex)

        ci = 1
        #drawn_clusters.sort(reverse=True)

        for c in self.phylogeny['subclone']:
            fill = self.phylogeny.loc[self.phylogeny['subclone'] == c]['color'].values[0]
            rc = draw.Rectangle(20, 25 * ci + height/2 + 100, 20, 25, fill=fill)
            dt = draw.Text(str(c), 12, x=6, y=25 * (ci + 1) + height/2 + 100, valign='top')
            container.append(rc)
            container.append(dt)
            ci += 1
        drawing.height = height+hmargin
        #ip.add_axes(drawing)
        return drawing
