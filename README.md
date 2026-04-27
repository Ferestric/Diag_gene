# Diag_gene
Diagnose cancer based on gene sequence

model.py - where the model architecture is defined
 - load_data():
    - read csv file
    - used one hot encoder to transform categorical variables & standard scaler to transform position to 0-1 range.
    - relabelled pathogenic/likely pathogenic --> 1, benign/likely benign --> 0
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
    --> We used parse_info to split strings into columns, then add CONSIG into row.
    
clinvar_subset.csv - small example subset from the list of chromosome 1 variants. Used to test out the model setup mostly.
