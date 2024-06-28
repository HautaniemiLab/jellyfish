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
    
    def build_total_graph(self, dropouts, rootid=0, plot=True):
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

        graph2 = Graph(directed=True)

        clonesdf = self.df.sort_values(['parent'])
        inferredsampledf = clonesdf.groupby(["subclone", "parent", "color"])['proportion'].sum().reset_index()
        print("inferredsampledf",inferredsampledf)
        # inferred samples are build separately, clones have proportions normalized by sum of proportions
        for index, row in inferredsampledf.iterrows():

            parent = int(row['parent'])
            if parent == -1:
                parent = 0

            samples = self.df.loc[self.df['subclone'] == row['subclone']]['sample']
            samples = ','.join(samples.to_list())
            if parent == -1:
                parent = 0

            c = graph2.add_vertex()
            color = row['color']
            # samples = self.df.loc[self.df['subclone'] == row['subclone']]['sample']
            # samples = ','.join(samples.to_list())
            c['id'] = samples
            c['label'] = samples
            c['subclone'] = int(row['subclone'])
            c['sample'] = samples
            #c['proportion'] = 1.0  # /(index+1)
            c['parent'] = parent
            c['color'] = color
            c['initialSize'] = 0
            c['proportion'] = row['proportion']
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
                graph2.add_edge(i1, i2)

            except Exception as e:
                print("Exception", e, parent)
                pass


        # add vertices for real samples
        for index, row in clonesdf.iterrows():

            parent = int(row['parent'])
            if parent == -1:
                parent = 0

            c = graph2.add_vertex()
            color = row['color']
            c["id"] = str(row['subclone']) + "_" + str(row['sample'])
            c["label"] = str(row['subclone']) + "_" + str(row['sample'])
            c["subclone"] = int(row['subclone'])
            c["sample"] = row['sample']
            c["proportion"] = row['proportion'] #1.0  # /(index+1)
            c['parent'] = parent
            c["color"] = color
            c["initialSize"] = 0
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

                    i1 = graph2.vs.find(subclone=vertex['subclone'], sample=vertex['sample'])
                    i2 = graph2.vs.find(subclone=parent, sample=vertex['sample'])

                    # if graph.es.find(i1.index,i2.index) == False:
                    # print("edge", i1, i2)
                    graph2.add_edge(i1, i2)

                except Exception as e:
                    print("Exception", e, parent)
                    pass

        for vertex in graph2.vs:

            # on the same phase
            parentsampleclone = graph2.vs.select(subclone=vertex['subclone'], site=vertex['site'],
                                                 rank=str(int(vertex['rank'])+1))
            if parentsampleclone:
                s1 = vertex
                s2 = parentsampleclone[0]
                # if graph2.es.find(s1.index,s2.index):
                graph2.add_edge(s1, s2)

            # between phases on the same site
            parentsampleclone = graph2.vs.select(subclone=vertex['subclone'], site=vertex['site'],
                                                 rank_gt=vertex['rank'])
            if parentsampleclone:
                s1 = vertex
                s2 = parentsampleclone[0]
                # if graph2.es.find(s1.index,s2.index):
                graph2.add_edge(s1, s2)

        connected_components = graph2.connected_components(mode="weak")

        for vertex in graph2.vs:
            # between phases from inferred root
            if vertex['site'] == "inferred":
                groupedsites = clonesdf.groupby('site')
                for gname, group in groupedsites:
                    minranked = group.sort_values('rank').head(1)
                    print(minranked['sample'].values[0])
                    childsampleclone = graph2.vs.select(subclone=vertex['subclone'], sample=minranked['sample'].values[0])
                    if childsampleclone:
                        s1 = vertex
                        s2 = childsampleclone[0]
                        print(s1)
                        print(s2)
                        # if graph2.es.find(s1.index,s2.index):
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
        ng = normalize_proportions(graph2, rootid)
        print("total_graph_norm", ng)
        if plot:
            igraph.plot(graph2, "./total_graph_un.pdf", centroid=(800,-800), bbox=(1600,1600), layout="sugiyama")
            igraph.plot(ng, "./total_graph_norm.pdf", centroid=(800,-800), bbox=(1600,1600), layout="sugiyama")

        return ng

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

    def build_phase_graph(self, dropouts, rootid=0):
        def normalize_proportions(g, rootid):
            # Get the root vertex
            root = g.vs.find(0)

            # Recursively normalize proportions
            def normalize_vertex(vertex):
                children = vertex.successors()
                # total_proportion = sum(child['proportion'] for child in children)
                if vertex['proportion'] <= 0:
                    vertex['proportion'] = 0.0
                for child in children:
                    # if len(children) < 2:
                    #     child['proportion'] = vertex['proportion'] - vertex['proportion'] / 4
                    #     print('build_graph_sep_sample', child)
                    # else:

                    if vertex['proportion'] < child['proportion'] and child['initialSize'] == 0:
                        vertex['proportion'] = child['proportion']

                    #print(' ', child)
                    normalize_vertex(child)

            normalize_vertex(root)
            return g

        graph2 = Graph(directed=True)

        dg = self.df.sort_values(['parent']).reset_index()

        # dg['proportion'] = dg['proportion'].clip(lower=0)
        dg['proportion'] = dg['proportion'] / dg['proportion'].sum()
        for index, row in dg.iterrows():

            parent = row['parent']
            rank = row['rank']
            #samplenum =
            purename = row['displayName']
            site = row['site']
            #sitenum =

            if parent == -1:
                parent = 0

            c = graph2.add_vertex()
            color = row['color']
            c['id'] = row['subclone']
            c['label'] = str(row['subclone']) + ' ' + row['sample']
            c['subclone'] = int(row['subclone'])
            c['sample'] = row['sample']
            c['proportion'] = 0.0 if row['proportion'] < 0.0 else row['proportion']  # /(index+1)
            c['parent'] = parent
            c['color'] = color
            c['initialSize'] = 0 if row['subclone'] in dropouts else 1
            c['proportion'] = row['proportion']
            c['rank'] = rank
            #c['samplenum'] = getSampleNum(row['sample'])
            c['purename'] = purename
            #c['sitenum'] = getSiteNum(row['sample'])
            c['site'] = site

        for vertex in graph2.vs:
            parent = int(vertex['parent'])
            if parent == -1:
                parent = 0

            if parent != 0:
                # connections inside sample
                try:
                    i1 = graph2.vs.find(subclone=parent, sample=vertex['sample'])
                    i2 = graph2.vs.find(subclone=vertex['subclone'], sample=vertex['sample'])
                    # if graph.es.find(i1.index,i2.index) == False:
                    graph2.add_edge(i1, i2)
                except Exception as e:
                    print('EXCEPTION build_phase_graph:', e, parent)
                    pass
                # connections to same sample site in following phase
                if int(vertex['rank']) >= 0:
                    # on the same phase
                    parentsampleclone = graph2.vs.select(subclone=vertex['subclone'], purename=vertex['purename'],
                                                         rank=int(vertex['rank'])-1)
                    if parentsampleclone:

                        s1 = parentsampleclone[0]
                        s2 = vertex
                        # if graph2.es.find(s1.index,s2.index):
                        graph2.add_edge(s1, s2)

                    # between phases on the same site
                    parentsampleclone = graph2.vs.select(subclone=vertex['subclone'], site=vertex['site'],
                                                         rank_lt=int(vertex['rank']))
                    if parentsampleclone:

                        s1 = parentsampleclone[0]
                        s2 = vertex
                        # if graph2.es.find(s1.index,s2.index):
                        graph2.add_edge(s1, s2)

            # if len(vertex.successors()) == 0 and len(vertex.predecessors()) == 0:
            #    graph2.delete_vertices(vertex)
        ng = normalize_proportions(graph2, rootid)
        # print('phasegraphnorm', ng)

        igraph.plot(ng, './phasegraph.pdf', bbox=(0, 0, 1200, 1200), layout='sugiyama')
        # igraph.plot(ng, './phasegraphnorm.pdf')

        return ng

