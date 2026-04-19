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

# --- read the VCF ---
records = []
with gzip.open('clinvar (1).vcf.gz', 'rt') as f:
    for line in f:
        if line.startswith('#'):
            continue  # skip header lines
        
        parts = line.strip().split('\t')
        chrom, pos, id_, ref, alt, qual, filter_, info = parts[:8]
        
        if str(chrom) != "1":
            break
        row = {
            'CHROM': chrom,
            'POS':   pos,
            'ID':    id_,
            'REF':   ref,
            'ALT':   alt,
        }
        # parse INFO and merge into row
        row.update(parse_info(info))
        records.append(row)

df = pd.DataFrame(records)
df.to_csv('clinvar_parsed.csv', index=False)