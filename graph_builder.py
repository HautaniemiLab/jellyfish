import re

import igraph
from igraph import Graph
import pandas

pandas.set_option('display.max_columns', None)


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

def getDepthRev(node):
    def fn(node):
        children = node.predecessors()
        depths = []
        for child in children:
            depth = fn(child)
            depths.append(depth)
        max_depth = max(depths) if depths else 0
        return max_depth + 1

    return fn(node)


def getNumNodes(node):
    node_cnt = [0]
    def fn(node):
        children = node.successors()
        depths = []
        for child in children:
            depth = fn(child)
            depths.append(depth)
            node_cnt[0] = node_cnt[0]+1
            return fn(child)

    fn(node)
    return node_cnt[0]


def getPhaseFromSampleName(name):
    if name[0] == 'p':
        return 1
    if name[0] == 'i':
        return 2
    if name[0] == 'r':
        return 3


def getSampleNum(name):
    if name[1].isnumeric():
        return int(name[1])
    else:
        return 1


def getSiteNum(name):
    if name[len(name) - 1].isnumeric():
        return int(name[len(name) - 1])
    else:
        return 0


def getSiteFromSampleName(sample):
    nums = r'[0-9]'
    phase_name = re.sub(nums, '', sample)
    return phase_name[1:]


def getPureSampleName(sample):
    nums = r'[0-9]'
    return re.sub(nums, '', sample)


class GraphBuilder:
    def __init__(self, patientdf):
        self.patientdf = patientdf

    def build_graph_sep(self, dropouts, rootid=1, plot=False):
        def normalize_fractions(g, rootid):
            # Get the root vertex
            try:
                root = g.vs[0]

                reducefrac = 1.0 / (len(g.vs)+1)

                # Recursively normalize normalize_fractions()
                def normalize_vertex(vertex):
                    children = vertex.successors()
                    # total_fraction = sum(child['fraction'] for child in children)

                    for child in children:
                        grandchildren = child.successors()
                        #     print("build_graph_sep", child)
                        if len(grandchildren) < 2 and len(children) == 1:
                            child['fraction'] = (1.0 - reducefrac / vertex['fraction'])  # - (vertex['fraction']*reducefrac)
                        else:
                            if len(grandchildren) == 0 and len(children) > 1:
                                child['fraction'] = (vertex['fraction'] / (len(children)+1)) #- reducefrac / vertex['fraction']
                            else:
                                child['fraction'] = (vertex['fraction'] / len(children)) + (len(children) * (reducefrac / 2))
                        if vertex['cluster'] == 1:
                            child['fraction'] = (vertex['fraction'] / len(children)) - reducefrac / 2

                        #     #child['fraction'] = child['fraction'] - child['fraction'] / (len(children) * len(children) * len(children))  # total_fracti
                        #     print("build_graph_sep", child)

                        # child['fraction'] = vertex['fraction'] - reducefrac #with treeToShapers()

                        # child['fraction'] = vertex['fraction'] / (len(children)) -reducefrac

                        # else:
                        #    child['fraction'] = vertex['fraction']-reducefrac
                        normalize_vertex(child)

                normalize_vertex(root)
            except Exception as ex:
                print(ex)
                pass

            return g

        graph2 = Graph(directed=True)

        dg = self.patientdf.sort_values(['parent'])
        dg = dg.groupby(["cluster", "parent", "color"])['frac'].sum().reset_index()

        for index, row in dg.iterrows():
            if row['cluster'] not in dropouts:
                parent = int(row['parent'])
                if parent == -1:
                    parent = 0

                c = graph2.add_vertex()
                color = row['color']
                samples = self.patientdf.loc[self.patientdf['cluster'] == row['cluster']]['sample']
                samples = ','.join(samples.to_list())
                c["id"] = row['cluster']
                c["label"] = str(int(row['cluster'])) + "\n" + str(samples)
                c["cluster"] = int(row['cluster'])
                c["sample"] = samples
                c["fraction"] = 1.0  # /(index+1)
                c['parent'] = parent
                c["color"] = color
                c["initialSize"] = 0
                c["frac"] = row['frac']

        for vertex in graph2.vs:
            if vertex['cluster'] not in dropouts:

                parent = int(vertex['parent'])
                if parent == -1:
                    parent = 0
                try:

                    i1 = graph2.vs.find(cluster=parent)
                    i2 = graph2.vs.find(cluster=vertex['cluster'])
                    # if graph.es.find(i1.index,i2.index) == False:
                    # print("edge", i1, i2)
                    graph2.add_edge(i1, i2)

                except Exception as e:
                    print("Exception", e, parent)
                    pass

            # Delete orphan vertice
            # if len(vertex.successors()) == 0 and len(vertex.predecessors()) == 0:
            #    graph2.delete_vertices(vertex)

            # print(graph2)

        print("rootgraphun", graph2)
        ng = normalize_fractions(graph2, rootid)
        print("rootgraphnorm", ng)
        if plot:
            igraph.plot(graph2, "./rootgraphun.pdf")
            igraph.plot(ng, "./rootgraphnorm.pdf")

        return ng

    def build_graph_sep_sample(self, dropouts, rootid=0, frac_threshold=0.0):
        def normalize_fractions(g, rootid, summa):
            # Get the root vertex
            root = g.vs.find(rootid)
            # rects = g.vs.select(initialSize=1)
            # rf_sum = 0
            # for r in rects:
            #     rf_sum += r['fraction']
            # Recursively normalize fractions
            def normalize_vertex(vertex):
                children = vertex.successors()
                # total_fraction = sum(child['fraction'] for child in children)

                if vertex['frac'] <= frac_threshold:
                    vertex['fraction'] = 0.0

                #vertex['fraction'] = vertex['fraction']/rf_sum
                numnewchild = 0
                newchildrenfrac = 0.0
                for child in children:
                    if child['initialSize'] == 0:
                        numnewchild += 1
                        newchildrenfrac += child['fraction']
                for child in children:

                    if vertex['fraction'] < newchildrenfrac and child['initialSize'] == 0:
                        #vertex['fraction'] = newchildrenfrac
                        vertex['fraction'] = vertex['fraction']+newchildrenfrac
                        if (vertex['fraction']-newchildrenfrac) < 0.01: # correction for finding tentacle attach point
                            vertex['fraction'] = vertex['fraction'] + 0.02
                    #print("build_graph_sep_sample", vertex)
                    #print("build_graph_sep_sample", child)
                    normalize_vertex(child)

            normalize_vertex(root)
            return g

        graph2 = Graph(directed=True)
        dg = self.patientdf.sort_values(['parent'])
        # dg = dg.groupby(["cluster","parent","color"])['frac'].sum().reset_index()
        #    #dg['frac'] = dg['frac']/dg['frac'].max()
        # dg['frac'] = dg['frac'].clip(lower=0)
        summa = dg['frac'].sum()
        dg['frac'] = dg['frac'] / dg['frac'].sum()
        #print("DG", dg)
        for index, row in dg.iterrows():
            parent = int(row['parent'])
            if parent == -1:
                parent = 0

            c = graph2.add_vertex()
            color = row['color']
            samples = self.patientdf.loc[self.patientdf['cluster'] == row['cluster']]['sample']
            samples = ','.join(samples.to_list())
            c["id"] = row['cluster']
            c["label"] = row['cluster']
            c["cluster"] = int(row['cluster'])
            c["sample"] = samples
            c["fraction"] = row['frac']  # /(index+1)
            c['parent'] = parent
            c["color"] = color
            c["initialSize"] = 0 if row['cluster'] in dropouts else 1
            c["frac"] = row['frac']

        for index, row in dg.iterrows():
            parent = int(row['parent'])
            if parent == -1:
                parent = 0

            try:
                if parent != 0:
                    i1 = graph2.vs.find(cluster=parent)
                    i2 = graph2.vs.find(cluster=row['cluster'])
                    # if graph.es.find(i1.index,i2.index) == False:
                    graph2.add_edge(i1, i2)

            except Exception as e:
                # Delete orphan vertice
                print("Exception", e)

                # pass
        # print(graph2)
        ng = normalize_fractions(graph2, rootid, summa)

        return ng

    def build_phase_graph(self, dropouts, rootid=0):
        def normalize_fractions(g, rootid):
            # Get the root vertex
            root = g.vs.find(0)

            # Recursively normalize fractions
            def normalize_vertex(vertex):
                children = vertex.successors()
                # total_fraction = sum(child['fraction'] for child in children)
                if vertex['frac'] <= 0:
                    vertex['fraction'] = 0.0
                for child in children:
                    # if len(children) < 2:
                    #     child['fraction'] = vertex['fraction'] - vertex['fraction'] / 4
                    #     print("build_graph_sep_sample", child)
                    # else:

                    if vertex['fraction'] < child['fraction'] and child['initialSize'] == 0:
                        vertex['fraction'] = child['fraction']

                    #print(" ", child)
                    normalize_vertex(child)

            normalize_vertex(root)
            return g

        graph2 = Graph(directed=True)

        dg = self.patientdf.sort_values(['parent']).reset_index()

        # dg['frac'] = dg['frac'].clip(lower=0)
        dg['frac'] = dg['frac'] / dg['frac'].sum()
        for index, row in dg.iterrows():

            parent = int(row['parent'])
            if parent == -1:
                parent = 0

            c = graph2.add_vertex()
            color = row['color']
            c["id"] = row['cluster']
            c["label"] = str(row['cluster']) + " " + row['sample']
            c["cluster"] = int(row['cluster'])
            c["sample"] = row['sample']
            c["fraction"] = 0.0 if row['frac'] < 0.0 else row['frac']  # /(index+1)
            c['parent'] = parent
            c["color"] = color
            c["initialSize"] = 0 if row['cluster'] in dropouts else 1
            c["frac"] = row['frac']
            c["phase"] = getPhaseFromSampleName(row['sample'])  # Encode 1 = p, 2 = i, 3 = r
            c["samplenum"] = getSampleNum(row['sample'])
            c["purename"] = getPureSampleName(row['sample'])
            c["sitenum"] = getSiteNum(row['sample'])
            c["site"] = getSiteFromSampleName(row['sample'])

        for vertex in graph2.vs:
            parent = int(vertex['parent'])
            if parent == -1:
                parent = 0

            if parent != 0:
                # connections inside sample
                try:
                    i1 = graph2.vs.find(cluster=parent, sample=vertex['sample'])
                    i2 = graph2.vs.find(cluster=vertex['cluster'], sample=vertex['sample'])
                    # if graph.es.find(i1.index,i2.index) == False:
                    graph2.add_edge(i1, i2)
                except Exception as e:
                    print("EXCEPTION:", e, parent)
                    pass
                # connections to same sample site in following phase
                if int(vertex['phase']) >= 0:
                    # on the same phase
                    parentsampleclone = graph2.vs.select(cluster=vertex['cluster'], purename=vertex['purename'],
                                                         phase=int(vertex['phase']),
                                                         samplenum=(int(vertex['samplenum']) - 1))
                    if parentsampleclone:

                        s1 = parentsampleclone[0]
                        s2 = vertex
                        # if graph2.es.find(s1.index,s2.index):
                        graph2.add_edge(s1, s2)

                    # between phases on the same site
                    parentsampleclone = graph2.vs.select(cluster=vertex['cluster'], site=vertex['site'],
                                                         samplenum=int(vertex['samplenum']),
                                                         phase_lt=int(vertex['phase']))
                    if parentsampleclone:

                        s1 = parentsampleclone[0]
                        s2 = vertex
                        # if graph2.es.find(s1.index,s2.index):
                        graph2.add_edge(s1, s2)

            # if len(vertex.successors()) == 0 and len(vertex.predecessors()) == 0:
            #    graph2.delete_vertices(vertex)
        ng = normalize_fractions(graph2, rootid)
        # print("phasegraphnorm", ng)

        igraph.plot(ng, "./phasegraph.pdf", bbox=(0, 0, 1200, 1200), layout="sugiyama")
        # igraph.plot(ng, "./phasegraphnorm.pdf")

        return ng

    def build_graph_sep_sample_norm_sum(self, dropouts, rootid=0):
        def normalize_fractions(g, rootid):
            # Get the root vertex
            root = g.vs.find(rootid)

            # Recursively normalize fractions
            def normalize_vertex(vertex):
                children = vertex.successors()
                # total_fraction = sum(child['fraction'] for child in children)

                for child in children:
                    # if len(children) < 2:
                    #     child['fraction'] = vertex['fraction']
                    #     print("build_graph_sep_sample", child)
                    # else:
                    if child['fraction'] > vertex['fraction']:
                        child['fraction'] = (vertex['fraction'] * child['fraction'])
                        # child['fraction'] = child['fraction'] - child['fraction'] / (
                        #        len(children) * len(children) * len(children))  # total_fraction

                    normalize_vertex(child)

            normalize_vertex(root)
            return g

        graph2 = Graph(directed=True)
        dg = self.patientdf.sort_values(['parent']).reset_index()
        # dg = dg.groupby(["cluster","parent","color"])['frac'].sum().reset_index()

        # dg.at[0,'frac'] = 1.0
        # print("rootfrac",rootfrac)
        if dg.at[0, 'frac'] < 0:
            dg.at[0, 'frac'] = -1 * dg.at[0, 'frac']
        #    dg.loc[dg['cluster'] == 1]['frac'] =-1*rootfrac
        #    rootfrac = -1*rootfrac
        # dg['frac'] = dg['frac'] / rootfrac

        dg['frac'] = dg['frac'] / dg['frac'].sum()

        #print("DG", dg)
        for index, row in dg.iterrows():
            parent = int(row['parent'])
            if parent == -1:
                parent = 0

            c = graph2.add_vertex()
            color = row['color']
            samples = self.patientdf.loc[self.patientdf['cluster'] == row['cluster']]['sample']
            samples = ','.join(samples.to_list())
            c["id"] = row['cluster']
            c["label"] = row['cluster']
            c["cluster"] = int(row['cluster'])
            c["sample"] = samples
            c["fraction"] = row['frac']  # /(index+1)
            c['parent'] = parent
            c["color"] = color
            c["initialSize"] = 0 if row['cluster'] in dropouts else 1
            c["frac"] = row['frac']

        for index, row in dg.iterrows():
            parent = int(row['parent'])
            if parent == -1:
                parent = 0

            try:
                if parent != 0:
                    i1 = graph2.vs.find(cluster=parent)
                    i2 = graph2.vs.find(cluster=row['cluster'])
                    # if graph.es.find(i1.index,i2.index) == False:
                    graph2.add_edge(i1, i2)

            except Exception as e:
                print("Exception", e)
                # pass
        # print(graph2)
        # ng = normalize_fractions(graph2, rootid)
        ng = graph2
        return ng

    def build_graph_per_sample(self, dropouts):

        graph = Graph(directed=True)

        for index, row in self.patientdf.iterrows():

            parent = int(row['parent'])
            if parent == -1:
                parent = 0

            c = graph.add_vertex()
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
            c["initialSize"] = 0 if row['cluster'] in dropouts else 1

        for index, row in self.patientdf.iterrows():
            parent = int(row['parent'])
            if parent == -1:
                parent = 0

            try:
                if parent != 0:
                    i1 = graph.vs.find(cluster=parent, sample=row['sample'])
                    i2 = graph.vs.find(cluster=row['cluster'], sample=row['sample'])
                    # if graph.es.find(i1.index,i2.index) == False:
                    if (i1.index, i2.index) not in graph.get_edgelist()[0:]:
                        graph.add_edge(i1, i2)

            except Exception as e:
                print(e)
                pass

        return graph
