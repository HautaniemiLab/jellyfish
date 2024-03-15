import argparse
import os
import pandas as pd
import svg_drawer


def main(**kwargs):
    # TODO: inferred samples
    # TODO: tentacle route optimization
    # TODO: Input data format/importers from other tools than clonevol eg. schism, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7447821/

    preproc_data_path = kwargs.get('path', 'data/preproc/')
    preproc_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(preproc_data_path) for f in filenames if f.endswith('.csv')]
    # preproc_files = ["data/preproc/test.csv"]
    for patientcsv in preproc_files:
        fnsplit = patientcsv.split('/')
        patient = fnsplit[len(fnsplit) - 1].split('.')[0]
        data = pd.read_csv(patientcsv, sep=",")
        data = data.drop(data.columns[0], axis=1).dropna(axis='rows')
        print(data)

        drawer = svg_drawer.Drawer(data, 0.000001, 0.9999999)
        jellyplot = drawer.draw(1.0, 1.0, patient)
        jellyplot.save_svg("./svg/" + patient + ".svg")
        jellyplot.save_png("./png/" + patient + ".png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help="Path to csv files with preprocessed clonal frequencies")
    args = parser.parse_args()
    main(**vars(args))
