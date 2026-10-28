from pathlib import Path
import numpy as np
import pandas as pd

RAW =Path("data/raw")
OUT= Path("data/clean.parquet")

# 1. Load all 8 CSVs. latin-1 because some labels contain a broken dash character.

frames=[]
for csv in sorted(RAW.glob("*.csv")):
    df=pd.read_csv(csv,encoding="latin-1",low_memory=False)
    print(f"{csv.name}: {len(df):,} rows")
    frames.append(df)
df=pd.concat(frames,ignore_index=True)

# 2. Column names have leading spaces (" Label"), so strip them.

df.columns =df.columns.str.strip()

# 3. "Fwd Header Length.1" is an exact duplicate of "Fwd Header Length".
df.drop(columns=["Fwd Header Length.1"])

# 4. Fix labels like "Web Attack \x96 Brute Force" -> "Web Attack - Brute Force"
df["Label"]=df["Label"].str.replace(r"[^\x00-\x7F]+","-",regex=True).str.strip()

# 5. Division by zero in CICFlowMeter produced inf. Treat inf as missing, then drop.

print("inf values :",np.isinf(df.select_dtypes("number")).sum().sum())
df=df.replace([np.inf, -np.inf], np.nan)
print("NaN rows   :",df.isna().any(axis=1).sum())
df=df.dropna()

# 6. Exact duplicate rows would leak between train and test sets later.

print("duplicates  :",df.duplicated().sum())
df=df.drop_duplicates()

print("n\Final shape:",df.shape)
print(df["Label"].value_counts())
df.to_parquet(OUT)