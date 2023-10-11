import os
import pathlib
import pandas as pd
import graph_builder
import sample_analyzer
import svg_drawer
import sys
import argparse


def main(**kwargs):
    clonevol_preproc_data_path = kwargs.get('clonevol_data', 'data/preproc/')
    #clonevol_preproc_data_path = kwargs.get('clonevol_data', 'data/newsamples/preproc/')
    clonevol_freq_data_path = kwargs.get('frequency_data', 'data/cellular_freqs/')
    #clonevol_freq_data_path = kwargs.get('frequency_data', 'data/newsamples/')
    mut_trees_file = kwargs.get('mutational_trees', 'data/j/mutTree_selected_models_20210311.csv')
    #mut_trees_file = kwargs.get('mutational_trees', 'data/newsamples/mutTree_selected_models.csv')
    models = pd.read_csv(mut_trees_file, sep='\t')
    files = list(pathlib.Path(clonevol_freq_data_path).rglob("*_cellular_freqs.csv"))
    #~/mnt/storageBig8/work/joikkone/evolution_s8/pyclone_2022/clonevol_outputs
    # files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(clonevol_freq_data_path) for f in filenames if f.endswith('_cellular_freqs.csv')]

    model_analyzer = sample_analyzer.DataAnalyzer(models, files)
    cfds = model_analyzer.calc_all_clonal_freqs()
    #preproc_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(clonevol_preproc_data_path) for f in filenames if f.endswith('.csv')]
    preproc_files = ["data/preproc/H030.csv"] #"data/preproc/OC005.csv", "data/preproc/H016.csv",
    for patientcsv in preproc_files:
        fnsplit = patientcsv.split('/')
        patient = fnsplit[len(fnsplit) - 1].split('.')[0]
        data = pd.read_csv(patientcsv, sep=",")
        data = data.drop(data.columns[0], axis=1).dropna(axis='rows')
        print(data)
        # "/Users/aimaaral/dev/clonevol/examples/" + patient + ".csv", sep=","
        #try:
        main_graph_builder = graph_builder.GraphBuilder(data)
        graph = main_graph_builder.build_graph_sep([],1,True)
        drawer = svg_drawer.Drawer(data, graph, 0.02, 0.99999, cfds)
        jellyplot = drawer.draw(1.0, 1.0, patient)
        jellyplot.save_svg("./svg/" + patient + ".svg")
        jellyplot.save_png("./png/" + patient + ".png")
            #print(jellyplot.as_svg())
        #except Exception as e:
        #    print("EXCEPTION:", patient, e)
        #    pass

args = argparse.ArgumentParser().parse_args()
main(**vars(args))
