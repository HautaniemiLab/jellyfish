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

                lap = el.args['lap']
                rap = el.args['rap']
                #print("TEEPEE",tpy, groupscaley)
                #return [Mx + float(t[0]), tpy * float(groupscaley)+float(t[1]/2), float(s[0]), (tpy * float(groupscaley)+float(t[1])/2)]
                if id.find('root') > 0:
                    return [(Mx + float(t[0]), lap * float(groupscaley)),
                            (Mx + float(t[0]), rap * float(groupscaley))]
                else:
                    return [(Mx + float(t[0]), lap*float(groupscaley)+float(t[1])), (Mx + float(t[0]), rap*float(groupscaley)+float(t[1]))]

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
            if child['proportion'] > 0:
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
    #fractions = [(float(n['proportion']) / float(node['proportion']) if float(node['proportion']) > 0.0 else 0.0) for n in childnodes]
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
    #print("pos",node['sample'],node['subclone'],positions)
    return positions

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


def draw_node(g, node, shaper, stacked_positions, spread_positions, translate, scale, inferred_scaler):
    sc = 100  # Segment count. Higher number produces smoother curves.

    firstSegment = 0
    for i in range(sc + 1):
        x = i / sc

        if shaper(x, 0) - shaper(x, 1) != 0:
            firstSegment = max(0, i - 1)
            break

    # p = svgwrite.path.Path()
    id_prefix = "clone_" + str(node['sample']) + "_"
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

    # st = stackTree(tree, shaper, 1)
    p.args['tp'] = 0.0
    l = p.args['d'].split('L')[1:]
    df = pd.DataFrame(l)
    df[['x', 'y']] = df[0].str.split(',', expand=True)

    maxy = float(df.max()['y'])
    miny = float(df.min()['y'])

    attach_pointy = (miny + maxy) / 2

    rtop = shaper(1, 0)
    rbottom = shaper(1, 1)
    rattach_pointy = rtop if (rbottom-rtop) > 0 else 1

    ltop = shaper(0, 0)
    lbottom = shaper(0, 1)
    lattach_pointy = ltop if (lbottom - ltop) > 0 else 1

    if node['site'] == 'inferred':
        p.args['lap'] = float(lattach_pointy)
        p.args['rap'] = float(rattach_pointy+1/(inferred_scaler*2))
    else:
        p.args['lap'] = float(lattach_pointy+node['fraction']/2)
        p.args['rap'] = float(rattach_pointy+node['fraction']/2) # (float(df.max()['y'])-float(df.min()['y']))/4
    g.append(p)



def addTreeToSvgGroupV1(tree: igraph.Graph, g, translate=[], scale=[], rootparent=0, inferred = False):
    total_depth = getDepth(tree.vs.find(parent=0))
    def process_node(node, parent_node=None, parent_shaper=lambda x, y: y, fractional_depth=0):

        def shaper(x, y):
            return parent_shaper(
                x,
                lerp(fancystep(fractional_depth, 1, x), 1, node['initialSize']) *
                node['proportion'] *
                (y - 0.5) + 0.5
            )

        childnodes = node.successors()
        # Children emerge as spread to better emphasize what their parent is
        spread_positions = stack_children(childnodes, node, True)
        # They end up as stacked to make the perception of the proportions easier
        stacked_positions = stack_children(childnodes, node, False)

        # Add current node to SVG group
        #if node['proportion'] > 0.001:

        draw_node(g, node, shaper, stacked_positions, spread_positions, translate, scale, total_depth)

        remaining_depth = getDepth(node)

        fractional_step = 0
        if node['initialSize'] == 0:
            fractional_step = (1 - fractional_depth) / (remaining_depth + 1)
        if parent_node:
            fractional_depth += fractional_step


        def make_interpolate_spread_stacked(child_idx):
            # Make an interpolator that smoothly interpolates between the spread and stacked positions
            def interpolator(x):
                if (node['initialSize'] == 0):
                    a = smoothstep(
                        fractional_depth + (1 - fractional_depth) / (remaining_depth + 1),
                        1,
                        x
                    )
                    s = 1 - spreadStrength
                    a = a * (1 - s) + s
                    return lerp(spread_positions[child_idx], stacked_positions[child_idx], a)
                else:
                    return stacked_positions[child_idx]
            return interpolator

        for i, child_node in enumerate(childnodes):
            interpolate_spread_stacked = make_interpolate_spread_stacked(i)

            process_node(
                child_node,
                node,
                lambda x, y: shaper(x, y + interpolate_spread_stacked(x)),
                fractional_depth
            )

    process_node(tree.vs.find(parent=rootparent))
    return g #scale_group_height(g, 1.0, scale[1], scale[0])

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
    p = None

    destination_sample = child['sample']
    source_sample = vertex['sample']
    #sampleboxpos = get_el_pos_of_group(sampleboxes[child['sample']])
    subclone = child['subclone']

    #left = int(sampleboxpos[0])

    endpos = get_el_pos_by_id(sampleboxes[destination_sample], drawing, "clone_" + str(destination_sample) + "_" + str(child['subclone']))[0]
    startpos = get_el_pos_by_id(sampleboxes[source_sample], drawing,
                                "clone_" + str(source_sample) + "_" + str(child['subclone']))[1]
    print('draw_tentacle',source_sample, child['subclone'], startpos)
    print('draw_tentacle',destination_sample, child['subclone'], endpos)
    if startpos == None or endpos == None:
        return None

    starty = float(startpos[1])
    endy = float(endpos[1]) - transY

    startx = startpos[0] + scalex
    if source_sample == "root":
        startx = startpos[0] + rw
    endx = endpos[0]

    p = draw.Path(id="tnt" + str(subclone) + "_" + source_sample + "_" + str(destination_sample), stroke_width=2,
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
        comp = self.composition #.reset_index()

        uniqsc = self.phylogeny['subclone'].unique()
        grpsamples = self.composition.groupby('sample')
        for sname, sample in grpsamples:
            subclones = sample['subclone'].tolist()
            for sc in uniqsc:
                if sc not in subclones:
                    s = pd.Series([sname, sc, 0.00], index=['sample', 'subclone', 'proportion'])
                    comp.loc[len(comp)]=s


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

        # initgraph = root_graph_builder.build_total_graph2(patient, [], frac_threshold, 0, True)
        #
        # subclone_cnts = joinedf.groupby('subclone')
        # for ind,grp in subclone_cnts:
        #     print("subclc",ind, len(grp))
        #     if len(grp) == 1:
        #         if ind != 1:
        #             dropouts.add(ind)
        #
        # i = 0
        # for index in initgraph.get_adjlist():
        #     if len(index) == 0:
        #         print("adjindex",i)
        #         endvs = initgraph.vs.find(i)
        #         dc = initgraph.vs.select(subclone=endvs['subclone'], site_ne='inferred', proportion_gt=0.0)
        #         if len(dc) < 2:
        #             dropouts.add(endvs['subclone'])
        #     i += 1
        #
        # dropouts.add(2)
        # dropouts.add(5)
        print("dropouts", dropouts)
        # Exclude root from dropouts
        if 1 in dropouts:
            dropouts.remove(1)

        root_graph_builder = graph_builder.GraphBuilder(joinedf)
        totalgraph = root_graph_builder.build_total_graph2(patient, dropouts, frac_threshold, 0, True)

        # Calculate dimensions by max number of samples in phases
        maxsamplesinphase = 0
        pgs = totalgraph.get_vertex_dataframe().reset_index().groupby(['rank'])
        for ind, gmr in pgs:
            if len(gmr['sample'].unique()) > maxsamplesinphase:
                maxsamplesinphase = len(gmr['sample'].unique())

        hmargin = maxsamplesinphase * 150
        height = maxsamplesinphase * 250 + hmargin
        width = 2500
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
        container = draw.Group(id='container', transform="translate(0," + str(transY) + ")")
        drawing.append(container)
        print('height', height)
        print('rooty',rootY)
        rootgroup = draw.Group(id='root',
                               transform="translate(0," + str(transY) + ") scale(" + str(rw) + "," + str(rh) + ")", x=0,
                               y=str(rootY), scaley=str(rh))

        rootnodes = totalgraph.vs.select(site="inferred")
        rootgraph = totalgraph.subgraph(rootnodes)

        rootjelly = addTreeToSvgGroupV1(rootgraph, rootgroup,[0, transY], [rw, rh], 0)
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

                    samplebox = addTreeToSvgGroupV1(samplegraph, sample_container, translate, [scalex, scaley], 0)

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
                print(vertex['sample'],vertex['subclone'],"=>",child['sample'],child['fraction'])
                if child['rank'] > vertex['rank']:

                    t = draw_tentacle(vertex, child, sampleboxes, drawing, transY, scalex, rw)
                    #if t and [child['sample'], int(child['subclone'])] not in drawn_tentacles:
                    if t:
                        container.append(t)
                    #    if [child['sample'], child['subclone']] not in drawn_tentacles:
                    #        drawn_tentacles.append([child['sample'], int(child['subclone'])])
                recursive_walk(child)

        rootvertex = totalgraph.vs.find(parent=0, site='inferred')
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
