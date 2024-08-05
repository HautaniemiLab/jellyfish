import argparse
import os
import pandas as pd
import svg_drawer
from pathlib import Path

def main(**kwargs):
    # TODO: inferred samples
    # TODO: tentacle route optimization
    # TODO: Input data format/importers from other tools than clonevol eg. schism, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7447821/

    #preproc_data_path = kwargs.get('path', './data/preproc/')
    #preproc_data_path = './data/preproc/'
    #preproc_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(preproc_data_path) for f in filenames if f.endswith('.csv')]
    path = Path("/home/aimaaral/dev/jellyfish/data/test/")
    ranks = pd.read_csv(path.joinpath("ranks.csv"))
    for p in Path(path).iterdir():
        if p.is_dir():
            composition = ""
            phylogeny = ""
            samples = ""
            patient = str(p.name)
            for file in p.iterdir():
                print(file.name)
                if file.name.endswith("compositions.tsv"):
                    composition = pd.read_csv(file, sep='\t')
                if file.name.endswith("phylogeny.tsv"):
                    phylogeny = pd.read_csv(file, sep='\t')
                if file.name.endswith("samples.tsv"):
                    samples = pd.read_csv(file, sep='\t')

            #data = data.drop(data.columns[0], axis=1).dropna(axis='rows')
            print(composition)
            drawer = svg_drawer.Drawer(samples, ranks, phylogeny, composition, 0.02, 0.9999999)
            jellyplot = drawer.draw(1.0, 1.0, patient)
            jellyplot.save_svg("./tmp/" + patient + ".svg")
            jellyplot.save_png("./tmp/" + patient + ".png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help="Path to csv files with preprocessed clonal frequencies")
    args = parser.parse_args()
    main(**vars(args))
