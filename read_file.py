import pandas as pd

csvfile = pd.read_csv("clinvar_parsed.csv")
field = "CLNSIG"
cond = csvfile[field]
conds = cond[(cond != "not_specified") & (cond != "not_provided") & (cond != "other")].value_counts()
print(conds.head(10))
