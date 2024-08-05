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

    def build_graph_sep(self, dropouts, rootid=1, plot=True):
        def normalize_proportions(g, rootid):
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
                        #     print('build_graph_sep', child)
                        if len(grandchildren) < 2 and len(children) == 1:
                            child['proportion'] = (1.0 - reducefrac / vertex['proportion'])  # - (vertex['proportion']*reducefrac)
                        else:
                            if len(grandchildren) == 0 and len(children) > 1:
                                child['proportion'] = (vertex['proportion'] / (len(children)+1)) #- reducefrac / vertex['proportion']
                            else:
                                child['proportion'] = (vertex['proportion'] / len(children)) + (len(children) * (reducefrac / 2))
                        if vertex['subclone'] == 1:
                            child['proportion'] = (vertex['proportion'] / len(children)) - reducefrac / 2

                        #     #child['proportion'] = child['proportion'] - child['proportion'] / (len(children) * len(children) * len(children))  # total_fracti
                        #     print('build_graph_sep', child)

                        # child['proportion'] = vertex['proportion'] - reducefrac #with treeToShapers()

                        # child['proportion'] = vertex['proportion'] / (len(children)) -reducefrac

                        # else:
                        #    child['proportion'] = vertex['proportion']-reducefrac
                        normalize_vertex(child)

                normalize_vertex(root)
            except Exception as ex:
                print(ex)
                pass

            return g

        graph2 = Graph(directed=True)

        dg = self.df.sort_values(['parent'])

        #dg = dg.groupby(["subclone", "parent", "color"])['proportion'].sum().reset_index()

        for index, row in dg.iterrows():
            if row['subclone'] not in dropouts:
                #print(row)
                parent = int(row['parent'])
                if parent == -1:
                    parent = 0

                c = graph2.add_vertex()
                color = row['color']
                #samples = self.df.loc[self.df['subclone'] == row['subclone']]['sample']
                #samples = ','.join(samples.to_list())
                c['id'] = str(row['sample'])+"_"+str(row['subclone'])
                c['label'] = str(row['sample'])+"_"+str(row['subclone'])
                c['subclone'] = int(row['subclone'])
                c['sample'] = row['sample']
                c['proportion'] = 1.0  # /(index+1)
                c['parent'] = parent
                c['color'] = color
                c['initialSize'] = 0
                c['proportion'] = row['proportion']
                c['rank'] = row['rank']
                c['purename'] = row['displayName']
                c['site'] = row['site']

        for vertex in graph2.vs:
            #print(vertex)
            if vertex['subclone'] not in dropouts:

                parent = int(vertex['parent'])
                if parent == -1:
                    parent = 0
                try:
                    parentrank = int(vertex['rank'])-1
                    if parentrank > 0:

                        candidate_parents = graph2.vs.select(rank_lt=int(vertex['rank']), subclone=parent, site=vertex['site'])
                        i1 = None
                        maxrank = 0
                        for v in candidate_parents:
                            if int(v['rank']) > maxrank:
                                maxrank = int(v['rank'])
                                i1 = v
                        if i1:
                            i2 = graph2.vs.select(subclone=vertex['subclone'])
                            # if graph.es.find(i1.index,i2.index) == False:
                            print('parent', i1)
                            print('child', i2[0])
                            graph2.add_edge(i1, i2[0])

                except Exception as e:
                    print('Exception build_graph_sep', e, parent)
                    pass


            # Delete orphan vertice
            # if len(vertex.successors()) == 0 and len(vertex.predecessors()) == 0:
            #    graph2.delete_vertices(vertex)

            # print(graph2)

        print('rootgraphun', graph2)
        for v in graph2.es:
            print(v)
        ng = normalize_proportions(graph2, rootid)
        print('rootgraphnorm', ng)

        if plot:
            igraph.plot(graph2, './rootgraphun.pdf')
            igraph.plot(ng, './rootgraphnorm.pdf')

        return ng
    
    def build_total_graph(self, dropouts, frac_threshold=0.0, rootid=0, plot=True):
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

                        #     #child['proportion'] = child['proportion'] - child['proportion'] / (len(children) * len(children) * len(children))  # total_fracti
                        #     print("build_graph_sep", child)

                        # child['proportion'] = vertex['proportion'] - reducefrac #with treeToShapers()

                        # child['proportion'] = vertex['proportion'] / (len(children)) -reducefrac

                        # else:
                        #    child['proportion'] = vertex['proportion']-reducefrac
                        normalize_vertex(child)

                normalize_vertex(root)
            except Exception as ex:
                print(ex)
                pass

            return g

        def normalize_proportions_real(g, rootid):
            # Get the root vertex
            root = g.vs.find(rootid)
            # rects = g.vs.select(initialSize=1)
            # rf_sum = 0
            # for r in rects:
            #     rf_sum += r['proportion']
            # Recursively normalize proportions
            def normalize_vertex(vertex):
                children = vertex.successors()
                # total_proportion = sum(child['proportion'] for child in children)

                if vertex['proportion'] <= frac_threshold:
                    vertex['proportion'] = 0.0

                #vertex['proportion'] = vertex['proportion']/rf_sum
                numnewchild = 0
                newchildrenfrac = 0.0
                for child in children:
                    if child['initialSize'] == 0:
                        numnewchild += 1
                        newchildrenfrac += child['proportion']
                for child in children:

                    if vertex['proportion'] < newchildrenfrac and child['initialSize'] == 0:
                        #vertex['proportion'] = newchildrenfrac
                        vertex['proportion'] = vertex['proportion']+newchildrenfrac
                        if (vertex['proportion']-newchildrenfrac) < 0.01: # correction for finding tentacle attach point
                            vertex['proportion'] = vertex['proportion'] + 0.02
                    #print('build_graph_sep_sample', vertex)
                    #print('build_graph_sep_sample', child)
                    normalize_vertex(child)

            normalize_vertex(root)
            return g

        graph2 = Graph(directed=True)

        clonesdf = self.df.sort_values(['parent'])
        #clonesdf['proportion'] = clonesdf['proportion'] / clonesdf['proportion'].sum()
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
                # samples = self.df.loc[self.df['subclone'] == row['subclone']]['sample']
                # samples = ','.join(samples.to_list())
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

                    # if graph.es.find(i1.index,i2.index) == False:
                    # print("edge", i1, i2)
                    if i1 != i2 and i2.degree(mode='in') < 2:
                        graph2.add_edge(i1, i2)

                except Exception as e:
                    print("Exception", e, parent)
                    pass

        for vertex in graph2.vs:

            # on the same phase
            childclone = graph2.vs.select(subclone=vertex['subclone'], site=vertex['site'],
                                                 rank=str(int(vertex['rank'])+1), site_ne="inferred")
            if childclone:
                s1 = vertex
                s2 = childclone[0]
                # if graph2.es.find(s1.index,s2.index):
                if s1!=s2 and s2.degree(mode='in') < 2:
                    graph2.add_edge(s1, s2)

            # between phases on the same site
            childclone = graph2.vs.select(subclone=vertex['subclone'], site=vertex['site'],
                                                 rank_gt=vertex['rank'], site_ne="inferred")
            if childclone:
                s1 = vertex
                s2 = childclone[0]
                # if graph2.es.find(s1.index,s2.index):
                if s1 != s2 and s2.degree(mode='in') < 2:
                    graph2.add_edge(s1, s2)

        connected_components = graph2.connected_components(mode="weak")

        for vertex in graph2.vs:
            # between phases from inferred root
            if vertex['site'] == "inferred":
                groupedsites = clonesdf.groupby('site')
                for gname, group in groupedsites:
                    minranked = group.sort_values('rank').head(1)
                    childsampleclone = graph2.vs.select(subclone=vertex['subclone'], sample=minranked['sample'].values[0], site_ne="inferred")
                    if childsampleclone:
                        s1 = vertex
                        s2 = childsampleclone[0]

                        # if graph2.es.find(s1.index,s2.index):
                        print('degree', s2.degree(mode='in'))
                        if s1 != s2 and s2.degree(mode='in') < 2:
                            graph2.add_edge(s1, s2)

                # for c in connected_components:
                #     for cc in c:
                #         clone = graph2.vs.find(cc)
                #         if clone['phase'] == 2 and clone['subclone'] == vertex['subclone']:
                #             s1 = vertex
                #             s2 = clone
                #             # if graph2.es.find(s1.index,s2.index):
                #             graph2.add_edge(s1, s2)
        # Delete orphan vertice
        # if len(vertex.successors()) == 0 and len(vertex.predecessors()) == 0:
        #    graph2.delete_vertices(vertex)

        # print(graph2)

        print("total_graph_un", graph2)
        #ng = normalize_proportions_inferred(graph2, rootid)
        #print("total_graph_norm", ng)
        if plot:
            igraph.plot(graph2, "./total_graph_un.pdf", centroid=(800,-800), bbox=(1600,1600), layout="sugiyama")
            #igraph.plot(ng, "./total_graph_norm.pdf", centroid=(800,-800), bbox=(1600,1600), layout="sugiyama")

        return graph2

    def build_graph_sep_sample(self, dropouts, rootid=0, frac_threshold=0.0):
        def normalize_proportions(g, rootid, summa):
            # Get the root vertex
            root = g.vs.find(rootid)
            # rects = g.vs.select(initialSize=1)
            # rf_sum = 0
            # for r in rects:
            #     rf_sum += r['proportion']
            # Recursively normalize proportions
            def normalize_vertex(vertex):
                children = vertex.successors()
                # total_proportion = sum(child['proportion'] for child in children)

                if vertex['proportion'] <= frac_threshold:
                    vertex['proportion'] = 0.0

                #vertex['proportion'] = vertex['proportion']/rf_sum
                numnewchild = 0
                newchildrenfrac = 0.0
                for child in children:
                    if child['initialSize'] == 0:
                        numnewchild += 1
                        newchildrenfrac += child['proportion']
                for child in children:

                    if vertex['proportion'] < newchildrenfrac and child['initialSize'] == 0:
                        #vertex['proportion'] = newchildrenfrac
                        vertex['proportion'] = vertex['proportion']+newchildrenfrac
                        if (vertex['proportion']-newchildrenfrac) < 0.01: # correction for finding tentacle attach point
                            vertex['proportion'] = vertex['proportion'] + 0.02
                    #print('build_graph_sep_sample', vertex)
                    #print('build_graph_sep_sample', child)
                    normalize_vertex(child)

            normalize_vertex(root)
            return g

        graph2 = Graph(directed=True)
        dg = self.df.sort_values(['subclone'])
        # dg = dg.groupby(['subclone''parent','color'])['proportion'].sum().reset_index()
        #    #dg['proportion'] = dg['proportion']/dg['proportion'].max()
        # dg['proportion'] = dg['proportion'].clip(lower=0)
        summa = dg['proportion'].sum()
        dg['proportion'] = dg['proportion'] / dg['proportion'].sum()
        #print('DG', dg)
        for index, row in dg.iterrows():
            parent = row['parent']
            if parent == -1:
                parent = 0

            c = graph2.add_vertex()
            color = row['color']
            samples = self.df.loc[self.df['subclone'] == row['subclone']]['sample']
            samples = ','.join(samples.to_list())
            c['id'] = row['subclone']
            c['label'] = row['subclone']
            c['subclone'] = int(row['subclone'])
            c['sample'] = samples
            c['proportion'] = row['proportion']  # /(index+1)
            c['parent'] = parent
            c['color'] = color
            c['initialSize'] = 0 if row['subclone'] in dropouts else 1
            c['proportion'] = row['proportion']
            c['purename'] = row['displayName']
            # c['sitenum'] = getSiteNum(row['sample'])
            c['site'] = row['site']

        for index, row in dg.iterrows():
            parent = row['parent']
            if parent == -1:
                parent = 0

            try:
                if parent != 0:
                    i1 = graph2.vs.find(subclone=parent)
                    i2 = graph2.vs.find(subclone=row['subclone'])
                    # if graph.es.find(i1.index,i2.index) == False:
                    graph2.add_edge(i1, i2)

            except Exception as e:
                # Delete orphan vertice
                print('Exception', e)

                # pass
        # print(graph2)
        ng = normalize_proportions(graph2, rootid, summa)

        return ng

