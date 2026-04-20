import gzip
import pandas as pd

def parse_info(info_str):
    """Parse a VCF INFO string into a dictionary."""
    info_dict = {}
    for field in info_str.split(';'):
        if '=' in field:
            key, value = field.split('=', 1)  # maxsplit=1 handles values with '='
            info_dict[key] = value
        else:
            # flag fields with no value (e.g. "DB", "COMMON")
            info_dict[field] = True
    return info_dict

# Target cancer-relevant chromosomes
CANCER_CHROMS = {'17', '13', '12', '7', '3', '8'}

# Clear labels (not uncertain significance or conflicting)
KEEP_SIGS = {
    'Benign', 'Likely_benign',
    'Pathogenic', 'Likely_pathogenic'
}

# --- read the VCF ---
records = []
with gzip.open('clinvar.vcf.gz', 'rt') as f:
    for line in f:
        if line.startswith('#'):
            continue  # skip header lines
        
        parts = line.strip().split('\t')

        chrom, pos, id_, ref, alt, qual, filter_, info = parts[:8]
        if chrom not in CANCER_CHROMS:
            continue
        
        info_dict = parse_info(info)

        if info_dict.get('CLNSIG') not in KEEP_SIGS: # skip unwanted significance labels
            continue
        
        if info_dict.get('CLNVC') != "single_nucleotide_variant": # skip non-single variants
            continue

        row = {
            'CHROM': chrom,
            'POS':   pos,
            'ID':    id_,
            'REF':   ref,
            'ALT':   alt
        }
        # parse INFO and merge into row
        row.update(info_dict)
        records.append(row)
        
df = pd.DataFrame(records)
df.to_csv('clinvar_parsed.csv', index=False)
print(df['CHROM'].value_counts())
print(df['CLNSIG'].value_counts())