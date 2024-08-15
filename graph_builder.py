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

            def normalize_vertex(vertex):
                children = vertex.successors()
                # total_proportion = sum(child['proportion'] for child in children)

                if vertex['site'] != 'inferred' and vertex['proportion'] <= frac_threshold:
                    vertex['proportion'] = 0.0

                # vertex['proportion'] = vertex['proportion']/rf_sum
                numnewchild = 0
                newchildrenfrac = 0.0
                for child in children:
                    if child['site'] != 'inferred' and child['initialSize'] == 0:
                        numnewchild += 1
                        newchildrenfrac += child['proportion']
                for child in children:

                    if vertex['site'] != 'inferred' and vertex['proportion'] < newchildrenfrac and child[
                        'initialSize'] == 0:
                        # vertex['proportion'] = newchildrenfrac
                        vertex['proportion'] = vertex['proportion'] + newchildrenfrac
                        if (vertex[
                                'proportion'] - newchildrenfrac) < 0.01:  # correction for finding tentacle attach point
                            vertex['proportion'] = vertex['proportion'] + 0.02

                    normalize_vertex(child)

            normalize_vertex(root)
            return g

        graph2 = Graph(directed=True)

        clonesdf = self.df.sort_values(['parent'])
        # clonesdf['proportion'] = clonesdf['proportion'] / clonesdf['proportion'].sum()
        inferredsampledf = self.df.groupby(["subclone", "parent", "color"])['proportion'].sum().reset_index()
        print("inferredsampledf", inferredsampledf)
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

        for vertex in graph2.vs:
            # if vertex['subclone'] not in dropouts:
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

        normalize_proportions_inferred(graph2, rootid)
        # add vertices for real samples
        samplegrps = self.df.groupby(["sample"])
        for sample_name, group in samplegrps:
            for index, row in inferredsampledf.iterrows():
                parent = int(row['parent'])
                if parent == -1:
                    parent = 0
                clone = self.df.loc[(self.df['sample'] == sample_name[0]) & (self.df['subclone'] == row['subclone'])]
                sample = self.df.loc[(self.df['sample'] == sample_name[0]) & (self.df['subclone'] == row['subclone'])]
                proportion = clone['proportion'].values[0] if not clone.empty else 0.0
                if proportion < frac_threshold:
                    proportion = 0.0

                if clone.empty:
                    clone = self.df.loc[self.df['subclone'] == row['subclone']].head(1)
                    sample = self.df.loc[self.df['sample'] == sample_name[0]].head(1)

                displayname = self.df.loc[self.df['sample'] == sample_name[0]].head(1)['displayName'].values[0]
                c = graph2.add_vertex()

                # rankgrp = self.df.groupby('rank')
                # for rank, rgrp in rankgrp:
                #     print("rank", rank, len(rgrp))
                #     clonegrp = rgrp.groupby('subclone')
                #     if rank == 1:
                #         clonegrp = rgrp.groupby('subclone')
                #         samplegrp = rgrp.groupby('sample')
                #         nsamples = samplegrp.count()
                #         if nsamples > 1:
                #             for clone, sgrp in clonegrp:
                #                 if len(sgrp) == 1:
                #                     initialSize = 0

                c["id"] = str(sample_name[0]) + "_" + str(row['subclone'])
                c["label"] = str(displayname) + "_" + str(row['subclone'])
                c["subclone"] = int(row['subclone'])
                c["sample"] = sample_name[0]
                c["proportion"] = proportion if proportion > 0.0 else 0.0  # 1.0  # /(index+1)
                c['parent'] = parent
                c["color"] = clone['color'].values[0]
                c["initialSize"] = 0 if row['subclone'] in dropouts else 1
                c["purename"] = displayname
                c["site"] = sample['site'].values[0]
                c["rank"] = sample['rank'].values[0]

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
                        if i1 != i2 and i2.degree(mode='in') < 1:
                            graph2.add_edge(i1, i2)

                    except Exception as e:
                        print("Exception", e, parent)
                        pass

            for vertex in graph2.vs:

                # on the same phase
                childclones = graph2.vs.select(subclone=vertex['subclone'], site=vertex['site'],
                                               rank=str(int(vertex['rank']) + 1), site_ne="inferred")
                for childclone in childclones:
                    s1 = vertex
                    s2 = childclone
                    # if graph2.es.find(s1.index,s2.index):
                    if s1 != s2 and s2.degree(mode='in') < 1:
                        graph2.add_edge(s1, s2)

                # between phases on the same site
                childclones = graph2.vs.select(subclone=vertex['subclone'], site=vertex['site'],
                                               rank_gt=vertex['rank'], site_ne="inferred")
                for childclone in childclones:
                    s1 = vertex
                    s2 = childclone
                    # if graph2.es.find(s1.index,s2.index):
                    if s1 != s2 and s2.degree(mode='in') < 1:
                        graph2.add_edge(s1, s2)

            # connected_components = graph2.connected_components(mode="weak")

            for vertex in graph2.vs:
                # between phases from inferred root
                if vertex['site'] == "inferred":
                    groupedsites = clonesdf.groupby('site')
                    for gname, group in groupedsites:
                        minrank = group['rank'].min()
                        minranked = group.loc[group['rank'] == minrank]['sample'].to_list()

                        # print("minranked",gname, minranked)
                        childclones = graph2.vs.select(subclone=vertex['subclone'], sample_in=minranked,
                                                       site_ne="inferred", initialSize=1)
                        for childclone in childclones:
                            s1 = vertex
                            s2 = childclone
                            # if graph2.es.find(s1.index,s2.index):
                            if s1 != s2 and s2.degree(mode='in') < 1:
                                graph2.add_edge(s1, s2)

        print("total_graph_un", graph2)
        # normalize_proportions_real(graph2, rootid)

        if plot:
            igraph.plot(graph2, "./total_graph_un_" + patient + ".pdf", centroid=(800, -800), bbox=(1600, 1600),
                        layout="sugiyama")
            # igraph.plot(ng, "./total_graph_norm.pdf", centroid=(800,-800), bbox=(1600,1600), layout="sugiyama")

        return graph2

    def build_total_graph2(self, patient, dropouts, frac_threshold=0.0, rootid=0, plot=True):
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

            def normalize_vertex(vertex):
                children = vertex.successors()
                # total_proportion = sum(child['proportion'] for child in children)

                if vertex['site'] != 'inferred' and vertex['proportion'] <= frac_threshold:
                    vertex['proportion'] = 0.0

                # vertex['proportion'] = vertex['proportion']/rf_sum
                numnewchild = 0
                newchildrenfrac = 0.0
                for child in children:
                    if child['site'] != 'inferred' and child['initialSize'] == 0:
                        numnewchild += 1
                        newchildrenfrac += child['proportion']
                for child in children:

                    if vertex['site'] != 'inferred' and vertex['proportion'] < newchildrenfrac and child[
                        'initialSize'] == 0:
                        # vertex['proportion'] = newchildrenfrac
                        vertex['proportion'] = vertex['proportion'] + newchildrenfrac
                        if (vertex[
                                'proportion'] - newchildrenfrac) < 0.01:  # correction for finding tentacle attach point
                            vertex['proportion'] = vertex['proportion'] + 0.02

                    normalize_vertex(child)

            normalize_vertex(root)
            return g

        graph2 = Graph(directed=True)

        clonesdf = self.df.sort_values(['parent'])
        # clonesdf['proportion'] = clonesdf['proportion'] / clonesdf['proportion'].sum()
        inferredsampledf = self.df.groupby(["subclone", "parent", "color"])['proportion'].sum().reset_index()
        print("inferredsampledf", inferredsampledf)
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
                #nfirstclones = len(self.df.loc[
                #                    (self.df['rank'] < 4) & (self.df['subclone'] == row['subclone']) & (
                #                                self.df['proportion'] > 0.0)])
                #nfirstranksamples = len(self.df.loc[self.df['rank'] < 4].groupby('sample'))
                #print(row['subclone'],nfirstclones, nfirstranksamples)
                #if nfirstclones >= nfirstranksamples:

                c = graph2.add_vertex()
                # samples = self.df.loc[self.df['subclone'] == row['subclone']]['sample']
                # samples = ','.join(samples.to_list())
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

        for vertex in graph2.vs:
            # if vertex['subclone'] not in dropouts:
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

        normalize_proportions_inferred(graph2, rootid)
        # add vertices for real samples
        samplegrps = self.df.groupby(["sample"])
        for sample_name, samplegroup in samplegrps:
            for ind, row in samplegroup.iterrows():
                parent = int(row['parent'])
                if parent == -1:
                    parent = 0

                proportion = row['proportion']
                if row['proportion'] < frac_threshold:
                    proportion = 0.0

                c = graph2.add_vertex()
                countprev = len(self.df.loc[(self.df['rank'] < row['rank']) & (self.df['subclone']==row['subclone']) & (self.df['proportion']>0.0)])

                # rankgrp = self.df.groupby('rank')
                # for rank, rgrp in rankgrp:
                #     print("rank", rank, len(rgrp))
                #     clonegrp = rgrp.groupby('subclone')
                #     if rank == 1:
                #         clonegrp = rgrp.groupby('subclone')
                #         samplegrp = rgrp.groupby('sample')
                #         nsamples = samplegrp.count()
                #         if nsamples > 1:
                #             for clone, sgrp in clonegrp:
                #                 if len(sgrp) == 1:
                #                     initialSize = 0
                print("adding",str(row['sample']) + "_" + str(row['subclone']))
                c["id"] = str(row['sample']) + "_" + str(row['subclone'])
                c["label"] = str(row['displayName']) + "_" + str(row['subclone'])
                c["subclone"] = int(row['subclone'])
                c["sample"] = row['sample']
                c["proportion"] = proportion if proportion > 0.0 else 0.0  # 1.0  # /(index+1)
                c['parent'] = parent
                c["color"] = row['color']
                c["initialSize"] = 0 if countprev < 1 else 1
                c["purename"] = row['displayName']
                c["site"] = row['site']
                c["rank"] = row['rank']

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
                        if i1 != i2 and i2.degree(mode='in') < 1:
                            graph2.add_edge(i1, i2)

                    except Exception as e:
                        print("Exception", e, parent)
                        pass

            for vertex in graph2.vs:

                # on the same phase
                childclones = graph2.vs.select(subclone=vertex['subclone'], site=vertex['site'],
                                               rank=str(int(vertex['rank']) + 1), site_ne="inferred")
                for childclone in childclones:
                    s1 = vertex
                    s2 = childclone
                    # if graph2.es.find(s1.index,s2.index):
                    if s1 != s2 and s2.degree(mode='in') < 1:
                        graph2.add_edge(s1, s2)

                # between phases on the same site
                childclones = graph2.vs.select(subclone=vertex['subclone'], site=vertex['site'],
                                               rank_gt=vertex['rank'], site_ne="inferred")
                for childclone in childclones:
                    s1 = vertex
                    s2 = childclone
                    # if graph2.es.find(s1.index,s2.index):
                    if s1 != s2 and s2.degree(mode='in') < 1:
                        graph2.add_edge(s1, s2)

            # connected_components = graph2.connected_components(mode="weak")

            for vertex in graph2.vs:
                # between phases from inferred root
                if vertex['site'] == "inferred":
                    groupedsites = clonesdf.groupby('site')
                    for gname, group in groupedsites:
                        minrank = group['rank'].min()
                        minranked = group.loc[group['rank'] == minrank]['sample'].to_list()

                        # print("minranked",gname, minranked)
                        childclones = graph2.vs.select(subclone=vertex['subclone'], sample_in=minranked,
                                                       site_ne="inferred", initialSize=1)
                        for childclone in childclones:
                            s1 = vertex
                            s2 = childclone
                            # if graph2.es.find(s1.index,s2.index):
                            if s1 != s2 and s2.degree(mode='in') < 1:
                                graph2.add_edge(s1, s2)

        print("total_graph_un", graph2)
        # normalize_proportions_real(graph2, rootid)

        if plot:
            igraph.plot(graph2, "./total_graph_un_" + patient + ".pdf", centroid=(800, -800), bbox=(1600, 1600),
                        layout="sugiyama")
            # igraph.plot(ng, "./total_graph_norm.pdf", centroid=(800,-800), bbox=(1600,1600), layout="sugiyama")

        return graph2