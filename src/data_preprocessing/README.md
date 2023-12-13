# Files for data gathering for training models

---

### Genomics models
The data for the genomics models is created by scraping all RefSeq genomes with at least 80% CheckM completeness and
less than 5% contamination. The data is then filtered to only include genomes with at least 1000 genes. The data is
then split into training, validation, and test sets with a 80/10/10 split, with a balanced number of genomes within each taxa.
Data can first be downloaded from: https://osf.io/btr3h

Run `dataset_2_fasta.py` to create fasta files for each genomic sequence.  
For genomes longer than 1500 bp (all of them), we splice into 1500 bp chunks for training in fasta files.
The fasta files are using conventional formatting, meaning a header line `>{sequence id} {index}' is followed by a line 
representing the sequence

Run `process_genomics_data.py` to create a mapping between a taxid and the phyla of a sequence.  There are our ground
labels for each sequence.  

