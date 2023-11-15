import argparse
import os
import pandas as pd
import svg_drawer


def main(**kwargs):
    clonevol_preproc_data_path = kwargs.get('clonevol_data', 'data/preproc/')
    #clonevol_preproc_data_path = kwargs.get('clonevol_data', 'data/newsamples/preproc/')

    preproc_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(clonevol_preproc_data_path) for f in filenames if f.endswith('.csv')]
    #preproc_files = ["data/preproc/H043.csv"] #"data/preproc/OC005.csv", "data/preproc/H016.csv",
    for patientcsv in preproc_files:
        fnsplit = patientcsv.split('/')
        patient = fnsplit[len(fnsplit) - 1].split('.')[0]
        data = pd.read_csv(patientcsv, sep=",")
        data = data.drop(data.columns[0], axis=1).dropna(axis='rows')
        print(data)
        # "/Users/aimaaral/dev/clonevol/examples/" + patient + ".csv", sep=","
        #try:
        #main_graph_builder = graph_builder.GraphBuilder(data)
        #graph = main_graph_builder.build_graph_sep([],1,True)

        drawer = svg_drawer.Drawer(data, 0.000001, 0.9999999)
        jellyplot = drawer.draw(1.0, 1.0, patient)
        jellyplot.save_svg("./svg/" + patient + ".svg")
        jellyplot.save_png("./png/" + patient + ".png")
            #print(jellyplot.as_svg())
        #except Exception as e:
        #    print("EXCEPTION:", patient, e)
        #    pass

args = argparse.ArgumentParser().parse_args()
main(**vars(args))
