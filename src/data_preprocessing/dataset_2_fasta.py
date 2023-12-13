"""
Converts the crude dataset to fasta format.
"""
import pandas as pd
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='parse dataset to fastas')
    parser.add_argument('dataset', help='directory containing test.tsv, df_1.tsv, classes.pkl')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dir = args.dataset

    df_test = pd.read_csv(dir + "test.tsv", sep="\t", index_col=None)
    df_train = pd.read_csv(dir + "df_1.tsv", sep="\t", index_col=None)
    for name, df in [("query", df_test), ("db_bert", df_train)]:
        with open(f"{os.path.join(dir, name)}.fa", "w") as outfile:
            for line in df.itertuples():
                outfile.write(f">{line.tax_id} {line.Index}\n")
                outfile.write(f"{line.x}\n")

if __name__ == '__main__':
    main()