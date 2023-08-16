import igraph
from igraph import Graph
import pandas

pandas.set_option('display.max_columns', None)


class GraphBuilder:
    def __init__(self, patientdf):
        self.patientdf = patientdf

    def build_graph_sep(self, dropouts=[], rootid=0, plot=False):
        def normalize_fractions(g, rootid):
            # Get the root vertex
            root = g.vs.find(rootid)

            # Recursively normalize normalize_fractions()
            def normalize_vertex(vertex):
                children = vertex.successors()
                # total_fraction = sum(child['fraction'] for child in children)

                for child in children:
                    if len(children) < 2:
                        child['fraction'] = vertex['fraction'] - vertex['fraction'] / 6
                        print("build_graph_sep", child)
                    else:
                        child['fraction'] = (vertex['fraction'] / len(children))
                        child['fraction'] = child['fraction'] - child['fraction'] / (
                                len(children) * len(children) * len(children))  # total_fracti
                        print("build_graph_sep", child)
                    normalize_vertex(child)

            normalize_vertex(root)

            return g

        graph2 = Graph(directed=True)

        dg = self.patientdf.sort_values(['parent']).reset_index()
        dg = dg.groupby(["cluster", "parent", "color"])['frac'].sum().reset_index()

        print(dg)
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
                c["label"] = row['cluster']
                c["cluster"] = int(row['cluster'])
                c["sample"] = samples
                c["fraction"] = 1.0  # /(index+1)
                c['parent'] = parent
                c["color"] = color
                c["initialSize"] = 0
                c["frac"] = row['frac']

        for index, row in dg.iterrows():
            if row['cluster'] not in dropouts:
                parent = int(row['parent'])
                if parent == -1:
                    parent = 0

                try:
                    if parent != 0:
                        i1 = graph2.vs.find(cluster=parent)
                        i2 = graph2.vs.find(cluster=row['cluster'])
                        # if graph.es.find(i1.index,i2.index) == False:
                        print("edge", i1, i2)
                        graph2.add_edge(i1, i2)

                except Exception as e:
                    print("Exception", e)
                    pass
            # print(graph2)
        print("unnormalizedgraph", graph2)
        ng = normalize_fractions(graph2, rootid)
        print("normalizedgraph", ng)
        if plot:
            igraph.plot(graph2, "./unnormalizedgraph.pdf")
            igraph.plot(ng, "./normalizedgraph.pdf")

        return ng

    def build_graph_sep_sample(self, dropouts, rootid=0):
        def normalize_fractions(g, rootid):
            # Get the root vertex
            root = g.vs.find(rootid)

            # Recursively normalize fractions
            def normalize_vertex(vertex):
                children = vertex.successors()
                # total_fraction = sum(child['fraction'] for child in children)

                for child in children:
                    if len(children) < 2:
                        child['fraction'] = vertex['fraction'] - vertex['fraction'] / 4
                        print("build_graph_sep_sample", child)
                    else:
                        child['fraction'] = (vertex['fraction'] / len(children))
                        #child['fraction'] = child['fraction'] - child['fraction'] / (
                        #        len(children) * len(children) * len(children))  # total_fraction

                        print("build_graph_sep_sample", child)
                    normalize_vertex(child)

            normalize_vertex(root)
            return g

        graph2 = Graph(directed=True)
        dg = self.patientdf.sort_values(['parent']).reset_index()
        # dg = dg.groupby(["cluster","parent","color"])['frac'].sum().reset_index()
        #    #dg['frac'] = dg['frac']/dg['frac'].max()

        print(dg)
        dg['frac'] = dg['frac'] / dg['frac'].max()
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
            c["fraction"] = 1.0  # /(index+1)
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
        ng = normalize_fractions(graph2, rootid)

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
                        #child['fraction'] = child['fraction'] - child['fraction'] / (
                        #        len(children) * len(children) * len(children))  # total_fraction

                    normalize_vertex(child)

            normalize_vertex(root)
            return g

        graph2 = Graph(directed=True)
        dg = self.patientdf.sort_values(['parent']).reset_index()
        # dg = dg.groupby(["cluster","parent","color"])['frac'].sum().reset_index()

        #dg.at[0,'frac'] = 1.0
        #print("rootfrac",rootfrac)
        if dg.at[0,'frac'] < 0:
            dg.at[0, 'frac'] = -1 * dg.at[0,'frac']
        #    dg.loc[dg['cluster'] == 1]['frac'] =-1*rootfrac
        #    rootfrac = -1*rootfrac
        #dg['frac'] = dg['frac'] / rootfrac

        dg['frac'] = dg['frac']/dg['frac'].sum()

        print("DG",dg)
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
        #ng = normalize_fractions(graph2, rootid)
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
                pass

        return graph
