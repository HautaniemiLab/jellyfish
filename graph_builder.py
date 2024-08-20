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


def set_initial_sizes(graph2, df, threshold):
    maxrank = df['rank'].max()
    minrank = df['rank'].min()
    subclones = df['subclone'].unique()
    for rank in range(maxrank,minrank-1,-1):
        print("ranker",rank)

        for s in subclones:
            if s == 1:
                continue
            emerged_earlier = df.loc[(df['rank'] < rank) & (df['subclone'] == s) & (df['proportion'] > 0.0)]
            if rank > minrank and len(emerged_earlier) == 0:
                emerges_now = graph2.vs.select(rank=rank, subclone=s)
                for v in emerges_now:
                    print("emerges", v['sample'], v['subclone'])
                    v['initialSize'] = 0
            if rank == minrank:
                in_current_rank = df.loc[(df['rank'] == rank) & (df['subclone'] == s) & (df['proportion'] > threshold)]
                print("in_current_rank", in_current_rank)
                if len(in_current_rank) == 1:
                    emerges_now = graph2.vs.select(sample=in_current_rank['sample'].values[0], subclone=s, proportion_gt=0)[0]
                    emerges_now['initialSize'] = 0
            #
            #         print("in_current_rank", emerges_now['sample'], emerges_now['subclone'])
            #         if s != 1:
            #             emerges_now['initialSize'] = 0

    emerged_vs = graph2.vs.select(site_ne='inferred', initialSize=0)
    for v in emerged_vs:
        inferredv = graph2.vs.find(site='inferred', subclone=v['subclone'])
        inferredv['proportion'] = 0.0


def has_connection(parent, child):
    for s in parent.successors():
        if s['sample'] == child['sample'] and s['subclone'] == child['subclone']:
            return True
    return False


class GraphBuilder:
    def __init__(self, df: pd.DataFrame):
        self.df = df


    def build_total_graph2(self, patient, dropouts, frac_threshold=0.0, rootid=0, plot=True):

        ''' Deprecated '''
        def normalize_proportions_inferred(g, rootid):
            # Get the root vertex
            try:
                root = g.vs.find(rootid)

                reducefrac = 1.0 / (len(g.vs) + 1)

                # Recursively normalize normalize_proportions()
                def normalize_vertex(vertex):
                    children = vertex.successors()
                    # total_proportion = sum(child['proportion'] for child in children)

                    for child in children:
                        grandchildren = child.successors()
                        #     print("build_graph_sep", child)
                        if len(grandchildren) < 2 and len(children) == 1:
                            child['proportion'] = (
                                    1.0 - reducefrac / vertex['proportion'])  # - (vertex['proportion']*reducefrac)
                        else:
                            if len(grandchildren) == 0 and len(children) > 1:
                                child['proportion'] = (vertex['proportion'] / (
                                        len(children) + 1))  # - reducefrac / vertex['proportion']
                            else:
                                child['proportion'] = (vertex['proportion'] / len(children)) + (
                                        len(children) * (reducefrac / 2))
                        if vertex['subclone'] == 1:
                            child['proportion'] = (vertex['proportion'] / len(children)) - reducefrac / 2

                        normalize_vertex(child)

                normalize_vertex(root)
            except Exception as ex:
                print(ex)
                pass

            return g

        def normalize_proportions_real(samplegraph, vs, rootid):
            # Get the root vertex
            root = None
            rootvs = samplegraph.vs.select(parent=rootid)

            if len(rootvs) > 0:
                root = rootvs[0]

            def calculate_total_fractions(vertex):
                sum_fraction = vertex['proportion']
                children = vertex.successors()
                for child in children:
                    sum_fraction += calculate_total_fractions(child)
                vertex['total_fraction'] = sum_fraction
                # assing value to original graph
                vs.find(id=vertex['id'])['total_fraction'] = sum_fraction

                return sum_fraction

            def normalize_children(parent, vertex):
                children = vertex.successors()
                for child in children:
                    normalize_children(vertex, child)
                vertex['proportion'] = vertex['total_fraction'] / parent['total_fraction'] if parent and parent['total_fraction']>0 else 1
                # assing value to original graph
                vs.find(id=vertex['id'])['proportion'] = vertex['proportion']

            calculate_total_fractions(root)
            normalize_children(None, root)
            print('root', root)
            for v in vs:
                print(v)

        graph2 = Graph(directed=True)

        clonesdf = self.df.sort_values(['parent'])
        # clonesdf['proportion'] = clonesdf['proportion'] / clonesdf['proportion'].sum()
        inferredsampledf = self.df.groupby(["subclone", "parent", "color"])['proportion'].sum().reset_index()
        print("inferredsampledf", inferredsampledf)
        # inferred samples are build separately, clones have proportions normalized by sum of proportions
        for index, row in inferredsampledf.iterrows():

            parent = int(row['parent'])
            if parent == -1:
                parent = 0

            samples = self.df.loc[self.df['subclone'] == row['subclone']]['sample']
            samples = ','.join(samples.to_list())

            c = graph2.add_vertex()
            c['id'] = samples
            c['label'] = "Inferred" + "\n" + samples
            c['subclone'] = int(row['subclone'])
            c['sample'] = "root"
            # c['proportion'] = 1.0  # /(index+1)
            c['parent'] = parent
            c["color"] = row['color']
            c['initialSize'] = 0
            c['proportion'] = 1.0
            c['rank'] = 0
            c['purename'] = "Inferred"
            c['site'] = "inferred"
            c["fraction"] = row['proportion']
            c["total_fraction"] = 0.0

        for vertex in graph2.vs:
            # inside the sample
            parent = int(vertex['parent'])
            if parent == -1:
                parent = 0
            try:

                i1 = graph2.vs.find(subclone=parent)
                i2 = graph2.vs.find(subclone=vertex['subclone'])
                # if graph.es.find(i1.index,i2.index) == False:
                # print("edge", i1, i2)
                if i1 != i2 and i2.degree(mode='in') < 2:
                    graph2.add_edge(i1, i2)

            except Exception as e:
                print("Exception", e, parent)
                pass

        #normalize_proportions_inferred(graph2, rootid)
        # add vertices for real samples
        samplegrps = self.df.groupby(["sample"])
        for sample_name, samplegroup in samplegrps:
            for ind, row in samplegroup.iterrows():

                parent = int(row['parent'])

                proportion = row['proportion']
                #if row['proportion'] < frac_threshold:
                #    proportion = 0.0

                if parent == -1:
                    parent = 0

                c = graph2.add_vertex()

                c["id"] = str(row['sample']) + "_" + str(row['subclone'])
                c["label"] = str(row['displayName']) + "_" + str(row['subclone'])
                c["subclone"] = int(row['subclone'])
                c["sample"] = row['sample']
                c["proportion"] = proportion if proportion > 0.0 else 0.0  # 1.0  # /(index+1)
                c['parent'] = parent
                c["color"] = row['color']
                c["initialSize"] = 1 #if countprev == 0 else 1
                if row['subclone'] == 1:
                    c["initialSize"] = 1

                c["purename"] = row['displayName']
                c["site"] = row['site']
                c["rank"] = row['rank']
                c["fraction"] = row['proportion']
                c["total_fraction"] = 0

            # Add edges inside the sample
        for vertex in graph2.vs.select(site_ne='inferred'):
            parent = int(vertex['parent'])

            try:
                i1 = graph2.vs.find(subclone=parent, sample=vertex['sample'], site_ne="inferred")
                i2 = graph2.vs.find(subclone=vertex['subclone'], sample=vertex['sample'], site_ne="inferred")

                # if graph.es.find(i1.index,i2.index) == False:
                # print("edge", i1, i2)
                if i1 != i2 and i2.degree(mode='in') < 1:
                    graph2.add_edge(i1, i2)

            except Exception as e:
                print("Exception", e, parent)
                pass

        set_initial_sizes(graph2, self.df, frac_threshold)

        # Add edges between the same site
        for vertex in graph2.vs.select(site_ne='inferred'):
            print(vertex['sample'],vertex['subclone'],vertex.index)

            # Find the next rank
            cc = graph2.vs.select(subclone=vertex['subclone'], site=vertex['site'],
                                  rank_gt=vertex['rank'])

            min_rank = self.df['rank'].max()
            for c in cc:
                if c['rank'] < min_rank:
                    min_rank = c['rank']

            childclones = graph2.vs.select(subclone=vertex['subclone'], site=vertex['site'],
                                           rank=min_rank, fraction_gt=frac_threshold, initialSize=1)
            for childclone in childclones:
                s1 = vertex
                s2 = childclone
                if s1 != s2:
                    graph2.add_edge(s1, s2)

        # Proportion normalization
        inferrednodes = graph2.vs.select(site="inferred")
        inferredgraph = graph2.subgraph(inferrednodes)
        normalize_proportions_real(inferredgraph, inferrednodes, 0)

        samplegrps = self.df.groupby(["sample"])
        for sample_name, samplegroup in samplegrps:
            samplenodes = graph2.vs.select(sample=sample_name[0])
            samplegraph = graph2.subgraph(samplenodes)

            normalize_proportions_real(samplegraph, samplenodes,0)

        for vertex in graph2.vs:
            # Add connections from inferred root
            if vertex['site'] == "inferred" and vertex['total_fraction'] > 0:

                #minrank = group['rank'].min()
                #minranked = group.loc[group['rank'] == minrank]['sample'].to_list()

                # print("minranked",gname, minranked)
                cc = graph2.vs.select(subclone=vertex['subclone'],
                                      site_ne="inferred", total_fraction_gt=0,initialSize=1)
                min_rank = self.df['rank'].max()

                for c in cc:
                    if c['rank'] < min_rank:
                        min_rank = c['rank']

                childclones = graph2.vs.select(subclone=vertex['subclone'],
                                               site_ne="inferred", rank=min_rank, total_fraction_gt=0, initialSize=1)
                for childclone in childclones:
                    s1 = vertex
                    s2 = childclone
                    print("succs",s1.successors())
                    print("s2",s2)

                    if not has_connection(s1,s2):
                        graph2.add_edge(s1, s2)


        if plot:
            igraph.plot(graph2, "./total_graph_un_" + patient + ".pdf", centroid=(800, -800), bbox=(1600, 1600),
                        layout="sugiyama")
            # igraph.plot(ng, "./total_graph_norm.pdf", centroid=(800,-800), bbox=(1600,1600), layout="sugiyama")

        return graph2

    def normalize_sample(self, sample_name):

        nodes = self.df.loc[self.df['sample']==sample_name]
        nodes['children'] = ""
        nodes['total_fraction'] = 0.0
        # Convert the table into a tree
        root = None
        for node in nodes.values():
            if node['parent'] > 0:
                #nodes[node['parent']]['children'].append(node)
                parent = nodes.loc[nodes['subclone']==node['parent']]
                parent['children'] = parent['children']+","+node['subclone']
            elif node['parent'] == -1:
                root = node

        # Function to calculate the sum of a node's fraction and its descendants' fractions
        def calculate_total_fractions(node):
            sum_fraction = node['proportion']

            for child in node['children'].str().split(','):
                sum_fraction += calculate_total_fractions(nodes.loc[nodes['subclone']==child])
            node['total_fraction'] = sum_fraction
            return sum_fraction

        # Function to normalize fractions so that the root node is 100%
        def normalize_children(parent, node):
            for child in node['children'].str().split(','):
                normalize_children(node, nodes.loc[nodes['subclone']==child])
            node['proportion'] = node['total_fraction'] / parent['total_fraction'] if parent else 1

        calculate_total_fractions(root)
        normalize_children(None, root)

        return root
