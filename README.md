# Diag_gene
Diagnose cancer based on gene sequence

model.py - where the model architecture is defined
read_file.py - made to check how many unique values there are in a certain column. For example, how many diagnosis is in CLNSIG
test.py - File to extract data from clinvar.vcf.gz
  - For simple columns such as:
            'CHROM': chrom,
            'POS':   pos,
            'ID':    id_,
            'REF':   ref,
            'ALT':   alt
    --> Just add their values into row
  - For INFO: They are big strings with a bunch of information.
    --> We used parse_info to split strings into columns, then take CONSIG
clinvar_subset.csv - small subset from the list of chromosome 1 variants. Used to test out the model setup mostly.
