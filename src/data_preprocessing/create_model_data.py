
import json
import pandas as pd


def intake_fasta_data(file_path: str) -> pd.DataFrame:
    """
    Intakes fasta file and returns a list of tuples containing the sequence id and the sequence itself
    Args:
        file_path (str): path to fasta file
    Returns:

    """
    with open(file_path, 'r') as f:
        data = [i.rstrip() for i in f.readlines()]
    fasta = []
    for idx in range(len(data)):
        if idx % 2 == 1:
            fasta.append({"id": data[idx - 1][1:],
                          "sequence": data[idx]})
    return pd.DataFrame(fasta)


def intake_json_data(file_path: str) -> pd.DataFrame:
    """
    Intakes json file consisting of a list of dictionaries
    with each dictionary containing a key "sequence" representing id
    and a key "caption" representing the taxa.
    Returns a dataframe containing the sequence id and the sequence itself
    Args:
        file_path (str): path to json file
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df.columns = ['id', 'caption']
    return df


# Join df_fasta and df_json
def merge_fasta_json(fasta_path:str,
                     json_path:str) -> pd.DataFrame:
    """
    Intakes fasta file of sequences with headers of taxids
    and a json file with keys "sequence" and "caption" representing
    the "{taxid} {index}" and the taxa itself.
    Returns a dataframe containing the taxa and the sequence
    """
    df_fasta = intake_fasta_data(fasta_path)
    df_json = intake_json_data(json_path)
    df = df_fasta.merge(df_json, on='id').drop('id', axis=1)
    return df


def get_label_mapper(df_1, df_2):
    """
    Returns a dictionary mapping labels to integers, using both the df_1 and test sets.
    """
    mapper = {}
    unique_labels_train = list(df_1['caption'].unique())
    unique_labels_test = list(df_2['caption'].unique())
    curr_label = 0
    for dataset in [unique_labels_train, unique_labels_test]:
        for label in dataset:
            if label not in mapper:
                mapper[label] = curr_label
                curr_label += 1
    return mapper

def convert_labels_to_int(df_1: pd.DataFrame,
                          df_2: pd.DataFrame):
    """
    Converts the labels in df_1 and df_2 to integers
    Args:
        df_1 (pd.DataFrame): dataframe containing the labels
        df_2 (pd.DataFrame): dataframe containing the labels
    Returns:
        df_1 and df_2 with the labels converted to integers
        mapping dictionary mapping labels to integers
    """
    mapper = get_label_mapper(df_1, df_2)
    df_1['label'] = df_1['caption'].map(mapper)
    df_2['label'] = df_2['caption'].map(mapper)
    return df_1, df_2, mapper

def main(fasta_path_1:str,
         seq_json_path_1:str,
         fasta_path_2:str,
         seq_json_path_2:str,
         output_path_1:str,
         output_path_2:str,
         int_labels:bool=False):
    """
    Intakes fasta file of sequence and a json file of the sequence id and the taxa.
    Outputs csv files of the sequence and label of interest.
    """
    df_1 = merge_fasta_json(fasta_path_1, seq_json_path_1)
    df_2 = merge_fasta_json(fasta_path_2, seq_json_path_2)
    if int_labels:
        df_1, df_2, mapper = convert_labels_to_int(df_1, df_2)
        df_1[['sequence', 'label']].to_csv(output_path_1, index=False)
        df_2[['sequence', 'label']].to_csv(output_path_2, index=False)
        with open(output_path_1 + "mapper" + '.json', 'w') as f:
            json.dump(mapper, f)
    else:
        df_1[['sequence', 'caption']].to_csv(output_path_1, index=False)
        df_2[['sequence', 'caption']].to_csv(output_path_2, index=False)
    return 0

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f1', '--fasta_path_1', help='path to fasta file 1')
    parser.add_argument('-s1', '--seq_json_path_1', help='path to json file 1')
    parser.add_argument('-f2', '--fasta_path_2', help='path to fasta file 2')
    parser.add_argument('-s2', '--seq_json_path_2', help='path to json file 2')
    parser.add_argument('-o1', '--output_path_1', help='path to output csv file 1')
    parser.add_argument('-o2', '--output_path_2', help='path to output csv file 2')
    parser.add_argument('-i', '--int_labels', help='whether to convert labels to integers', action='store_true')
    args = parser.parse_args()
    main(args.fasta_path_1,
         args.seq_json_path_1,
         args.fasta_path_2,
         args.seq_json_path_2,
         args.output_path_1,
         args.output_path_2,
         args.int_labels)