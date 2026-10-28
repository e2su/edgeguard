from pathlib import Path
import pandas as pd 

IN = Path("data/clean.parquet")
OUT = Path("data/perpared.parquet")

# Decision 1: original label -> attack family. Labels not listed here are dropped.
FAMILY = {
    "BENIGN": "BENIGN",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "DDoS": "DoS",
    "PortScan": "PortScan",
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    "Web Attack - Brute Force": "WebAttack",
    "Web Attack - XSS": "WebAttack",
    "Web Attack - Sql Injection": "WebAttack",
}

# Decision 2: only features the Pi can compute the same way from live packets.
FEATURES = [
    # size
    "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min",
    "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std",
    # rate
    "Flow Bytes/s", "Flow Packets/s",
    # timing (microseconds)
    "Flow Duration",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Mean", "Bwd IAT Mean",
    # TCP flags
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count",
    # TCP window size of the first packet in each direction
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
]


df=pd.read_parquet(IN)
print("Loaded:",df.shape)

# Map labels to families; unmapped labels (Bot, Infiltration, Heartbleed) become NaN
df["Family"] = df["Label"].map(FAMILY)
print("Dropped (Bot/Infiltration/Heartbleed):",
      df["Family"].isna().sum())
df = df.dropna(subset=["Family"])

# Keep only our features + the target
df=df[FEATURES + ["Family"]]

# With fewer columns, some rows that were different become identical again
print("New duplicates after selecting features:",
      df.duplicated().sum())
df= df.drop_duplicates()

# Same features but different family = rows the model can never get right
conflicts=df[df.duplicated(subset=FEATURES ,keep=False)]
print("Conlficting rows (same features , different familiy:)",
      len(conflicts))
df=df.drop_duplicates(FEATURES ,keep=False)

print("\nFinal shape:",df.shape)
print(df["Family"].value_counts())
df.to_parquet(OUT)