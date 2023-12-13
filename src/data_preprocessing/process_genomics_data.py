"""
Process genomics data
"""
import argparse

import csv
import os
# import ray
# ray.init()
from ete3 import NCBITaxa
import json
# import modin.pandas as pd
import pandas as pd
from typing import List

# initialize ray for modin
ncbi = NCBITaxa()


def get_desired_ranks(taxid, desired_ranks):
    """
    Given a taxid and a list of desired ranks, return a dictionary of the ranks and their corresponding taxids.
    Extracts information from the NCBI taxonomy database.
    Args:
        taxid: the taxid of the taxa
        desired_ranks: the desired ranks of the taxa
    Returns:
        a dictionary of the ranks and their corresponding taxids
    """
    lineage = ncbi.get_lineage(taxid)
    names = ncbi.get_taxid_translator(lineage)
    lineage2ranks = ncbi.get_rank(names)
    ranks2lineage = dict((rank, taxid) for (taxid, rank) in lineage2ranks.items())
    return {'{}_id'.format(rank): ranks2lineage.get(rank, '<not present>') for rank in desired_ranks}


def get_phyla(taxid):
    """
    Get the phylum of the taxa given a child hiearchy taxids
    Args:
        taxid: the taxid of the taxa
    Returns:
        the NCBI phylum of the taxa
    """
    return get_desired_ranks(taxid, ['phylum'])['phylum_id']


def get_name_taxa(taxid):
    """
    Get the name of the taxa given a taxid
    Args:
        taxid: the taxid of the taxa
    Returns:
        the NCBI name of the taxa
    """
    if taxid == '<not present>':
        return 'unclassified'
    else:
        return ncbi.get_taxid_translator([taxid])[int(taxid)]


def get_phyla_info_df(taxids: List[str]) -> pd.DataFrame:
    """
    Given a list of entries with "<taxid> <frag_index>" format, return a dataframe with the following columns:
    - taxid
    - frag_index
    - phylum_id
    - phylum_name
    """
    df = pd.DataFrame(taxids, columns=['initial'])
    df['taxid'] = df['initial'].map(lambda x: x.split(' ')[0])
    df['index'] = df['initial'].map(lambda x: x.split(' ')[1])
    df['phylum_id'] = df['taxid'].map(lambda x: get_phyla(x))
    df['caption'] = df['phylum_id'].map(lambda x: get_name_taxa(x))
    df['sequence'] = df['initial'].map(lambda x: '_'.join(x.split(' ')))
    return df


def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input fasta file', required=True)
    parser.add_argument('-o', '--output', help='output name', required=True)
    return parser.parse_args()


def read_fasta(input_fasta):
    with open(input_fasta, 'r') as f:
        lines = [line.rstrip() for line in f.readlines()]
    return [line[1:] for line in lines[::2]]


def main():
    # Use an argparser and get the input fasta file
    args = parse_args()
    input_fasta = args.input
    output = args.output
    # Read in the fasta file and get the taxids
    data = get_phyla_info_df(read_fasta(input_fasta))



if __name__ == '__main__':
    main()
