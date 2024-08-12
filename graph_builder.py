import re

import igraph
from igraph import Graph
import pandas as pd

pd.set_option('display.max_columns', None)


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

class GraphBuilder:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def build_total_graph(self, patient, dropouts, frac_threshold=0.0, rootid=0, plot=True):
        def normalize_proportions_inferred(g, rootid):
            # Get the root vertex
            try:
                root = g.vs[0]

                reducefrac = 1.0 / (len(g.vs)+1)

                # Recursively normalize normalize_proportions()
                def normalize_vertex(vertex):
                    children = vertex.successors()
                    # total_proportion = sum(child['proportion'] for child in children)

                    for child in children:
                        grandchildren = child.successors()
                        #     print("build_graph_sep", child)
                        if len(grandchildren) < 2 and len(children) == 1:
                            child['proportion'] = (1.0 - reducefrac / vertex['proportion'])  # - (vertex['proportion']*reducefrac)
                        else:
                            if len(grandchildren) == 0 and len(children) > 1:
                                child['proportion'] = (vertex['proportion'] / (len(children)+1)) #- reducefrac / vertex['proportion']
                            else:
                                child['proportion'] = (vertex['proportion'] / len(children)) + (len(children) * (reducefrac / 2))
                        if vertex['subclone'] == 1:
                            child['proportion'] = (vertex['proportion'] / len(children)) - reducefrac / 2

                        normalize_vertex(child)

                normalize_vertex(root)
            except Exception as ex:
                print(ex)
                pass

            return g

        graph2 = Graph(directed=True)

        clonesdf = self.df.sort_values(['parent'])
        inferredsampledf = self.df.groupby(["subclone", "parent", "color"])['proportion'].sum().reset_index()
        print("inferredsampledf",inferredsampledf)
        # inferred samples are build separately, clones have proportions normalized by sum of proportions
        for index, row in inferredsampledf.iterrows():
            if row['subclone'] not in dropouts:
                parent = int(row['parent'])
                if parent == -1:
                    parent = 0

                samples = self.df.loc[self.df['subclone'] == row['subclone']]['sample']
                samples = ','.join(samples.to_list())
                if parent == -1:
                    parent = 0

                c = graph2.add_vertex()
                c['id'] = samples
                c['label'] = "Inferred"+ "\n" + samples
                c['subclone'] = int(row['subclone'])
                c['sample'] = "root"
                #c['proportion'] = 1.0  # /(index+1)
                c['parent'] = parent
                c["color"] = row['color']
                c['initialSize'] = 0
                c['proportion'] = 1.0
                c['rank'] = 0
                c['purename'] = "Inferred"
                c['site'] = "inferred"

        for vertex in graph2.vs:
            #if vertex['subclone'] not in dropouts:
            # inside the sample
            parent = int(vertex['parent'])
            if parent == -1:
                parent = 0
            try:

                i1 = graph2.vs.find(subclone=parent)
                i2 = graph2.vs.find(subclone=vertex['subclone'])
                # if graph.es.find(i1.index,i2.index) == False:
                # print("edge", i1, i2)
                if i1!=i2 and i2.degree(mode='in') < 2:
                    graph2.add_edge(i1, i2)

            except Exception as e:
                print("Exception", e, parent)
                pass

        normalize_proportions_inferred(graph2, rootid)
        # add vertices for real samples

        for index, row in clonesdf.iterrows():

            parent = int(row['parent'])
            if parent == -1:
                parent = 0

            c = graph2.add_vertex()

            c["id"] = str(row['sample']) + "_" + str(row['subclone'])
            c["label"] = str(row['sample']) + "_" + str(row['subclone'])
            c["subclone"] = int(row['subclone'])
            c["sample"] = row['sample']
            c["proportion"] = row['proportion'] if row['proportion'] > 0.0 else 0.0 #1.0  # /(index+1)
            c['parent'] = parent
            c["color"] = row['color']
            c["initialSize"] = 0 if row['subclone'] in dropouts else 1
            c["purename"] = row['displayName']
            c["site"] = row['site']
            c["rank"] = int(row['rank'])

        for vertex in graph2.vs:
            if vertex['site'] != "inferred":
                # inside the sample
                parent = int(vertex['parent'])
                if parent == -1:
                    parent = 0
                try:
                    i1 = graph2.vs.find(subclone=parent, sample=vertex['sample'], site_ne="inferred")
                    i2 = graph2.vs.find(subclone=vertex['subclone'], sample=vertex['sample'], site_ne="inferred")

                    if i1 != i2 and i2.degree(mode='in') < 2:
                        graph2.add_edge(i1, i2)

                except Exception as e:
                    print("Exception", e, parent)
                    pass

        for vertex in graph2.vs:

            # on the same phase
            childclones = graph2.vs.select(subclone=vertex['subclone'], site=vertex['site'],
                                                 rank=str(int(vertex['rank'])+1), site_ne="inferred")
            for childclone in childclones:
                s1 = vertex
                s2 = childclone
                # if graph2.es.find(s1.index,s2.index):
                if s1!=s2 and s2.degree(mode='in') < 2:
                    graph2.add_edge(s1, s2)

            # between phases on the same site
            childclones = graph2.vs.select(subclone=vertex['subclone'], site=vertex['site'],
                                                 rank_gt=vertex['rank'], site_ne="inferred")
            for childclone in childclones:
                s1 = vertex
                s2 = childclone
                # if graph2.es.find(s1.index,s2.index):
                if s1 != s2 and s2.degree(mode='in') < 2:
                    graph2.add_edge(s1, s2)

        #connected_components = graph2.connected_components(mode="weak")

        for vertex in graph2.vs:
            # between phases from inferred root
            if vertex['site'] == "inferred":
                groupedsites = clonesdf.groupby('site')
                for gname, group in groupedsites:
                    minrank = group['rank'].min()
                    minranked = group.loc[group['rank']==minrank]['sample'].to_list()

                    print("minranked",gname, minranked)
                    childclones = graph2.vs.select(subclone=vertex['subclone'], sample_in=minranked, site_ne="inferred")
                    for childclone in childclones:
                        s1 = vertex
                        s2 = childclone
                        # if graph2.es.find(s1.index,s2.index):
                        if s1 != s2 and s2.degree(mode='in') < 2:
                            graph2.add_edge(s1, s2)

        if plot:
            igraph.plot(graph2, "./total_graph_un_"+patient+".pdf", centroid=(800,-800), bbox=(1600,1600), layout="star")
            #igraph.plot(ng, "./total_graph_norm.pdf", centroid=(800,-800), bbox=(1600,1600), layout="sugiyama")

        return graph2

