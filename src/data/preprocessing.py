from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils.io import read_csv_maybe_sample


ATTACK_CATEGORY_MAP = {
    "DDoS-RSTFINFlood": "DDoS",
    "DDoS-PSHACK_Flood": "DDoS",
    "DDoS-SYN_Flood": "DDoS",
    "DDoS-UDP_Flood": "DDoS",
    "DDoS-TCP_Flood": "DDoS",
    "DDoS-ICMP_Flood": "DDoS",
    "DDoS-SynonymousIP_Flood": "DDoS",
    "DDoS-ACK_Fragmentation": "DDoS",
    "DDoS-UDP_Fragmentation": "DDoS",
    "DDoS-ICMP_Fragmentation": "DDoS",
    "DDoS-SlowLoris": "DDoS",
    "DDoS-HTTP_Flood": "DDoS",
    "DoS-UDP_Flood": "DoS",
    "DoS-TCP_Flood": "DoS",
    "DoS-SYN_Flood": "DoS",
    "DoS-HTTP_Flood": "DoS",
    "Mirai-greeth_flood": "Mirai",
    "Mirai-greip_flood": "Mirai",
    "Mirai-udpplain": "Mirai",
    "Recon-HostDiscovery": "Recon",
    "Recon-OSScan": "Recon",
    "Recon-PortScan": "Recon",
    "Recon-PingSweep": "Recon",
    "Recon-VulnerabilityScan": "Recon",
    "VulnerabilityScan": "Recon",
    "DNS_Spoofing": "Spoofing",
    "MITM-ArpSpoofing": "Spoofing",
    "Spoofing-ARP": "Spoofing",
    "Spoofing-DNS": "Spoofing",
    "BrowserHijacking": "Web",
    "Backdoor_Malware": "Web",
    "XSS": "Web",
    "Uploading_Attack": "Web",
    "SqlInjection": "Web",
    "CommandInjection": "Web",
    "Web-Based": "Web",
    "DictionaryBruteForce": "BruteForce",
    "BenignTraffic": "Benign",
    "BENIGN": "Benign",
    "Benign": "Benign",
    "benign": "Benign",
    "normal": "Benign",
}

LABEL_CANDIDATES = [
    "Label",
    "label",
    "Class",
    "class",
    "Attack",
    "attack",
    "Attack_type",
    "attack_type",
    "detailed-label",
    "detailed_label",
    "category",
    "Category",
    "target",
    "Target",
    "type",
    "Type",
]

LEAKAGE_PATTERNS = [
    r"\btimestamp\b",
    r"\bdatetime\b",
    r"\bdate\b",
    r"\bflow[\s_\-]?id\b",
    r"\bid\b",
    r"\bsrc[\s_\-]?ip\b",
    r"\bdst[\s_\-]?ip\b",
    r"\bsource[\s_\-]?ip\b",
    r"\bdestination[\s_\-]?ip\b",
    r"\bsrc[\s_\-]?mac\b",
    r"\bdst[\s_\-]?mac\b",
    r"\bmac\b",
    r"\bip\b",
    r"\bcapture\b",
    r"\bpcap\b",
    r"\bfile\b",
    r"\bfilename\b",
]

EDGE_LABEL_COL = "EdgeLabel"
EDGE_LEAKAGE_DROP_COLUMNS = [
    "arp_dst_proto_ipv4",
    "arp_src_proto_ipv4",
    "frame_time",
    "http_file_data",
    "http_referer",
    "http_request_full_uri",
    "http_request_uri_query",
    "http_tls_port",
    "icmp_transmit_timestamp",
    "ip_dst_host",
    "ip_src_host",
    "mqtt_msg",
    "mqtt_msg_decoded_as",
    "mqtt_msgtype",
    "mqtt_topic",
    "mqtt_topic_len",
    "tcp_dstport",
    "tcp_options",
    "tcp_payload",
    "tcp_srcport",
    "udp_port",
    "udp_stream",
    "udp_time_delta",
]


def make_synthetic_ciciot_demo(output_path, n_per_class=12000, seed=42):
    rng = np.random.default_rng(seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_names = [
        "flow_duration",
        "fwd_pkts_tot",
        "bwd_pkts_tot",
        "fwd_data_pkts_tot",
        "bwd_data_pkts_tot",
        "fwd_pkts_per_sec",
        "bwd_pkts_per_sec",
        "flow_pkts_per_sec",
        "fwd_byts_b_avg",
        "bwd_byts_b_avg",
        "fwd_iat_mean",
        "bwd_iat_mean",
        "flow_iat_mean",
        "pkt_len_mean",
        "pkt_len_std",
    ]
    specs = [
        ("BenignTraffic", 0.5, 0.35),
        ("DDoS-TCP_Flood", 1.2, 0.55),
        ("DoS-SYN_Flood", 1.0, 0.55),
        ("Mirai-udpplain", 1.5, 0.70),
        ("Recon-PortScan", 0.9, 0.45),
        ("Spoofing-ARP", 0.8, 0.50),
        ("SqlInjection", 1.1, 0.60),
        ("DictionaryBruteForce", 1.3, 0.70),
    ]
    frames = []
    for label, loc, scale in specs:
        x = rng.normal(loc=loc, scale=scale, size=(n_per_class, len(feature_names)))
        df = pd.DataFrame(x, columns=feature_names)
        df["flow_pkts_per_sec"] = np.abs(df["flow_pkts_per_sec"] * (1.0 + loc))
        df["fwd_pkts_tot"] = np.abs(df["fwd_pkts_tot"] * 10).round()
        df["bwd_pkts_tot"] = np.abs(df["bwd_pkts_tot"] * 8).round()
        df["Label"] = label
        frames.append(df)
    out = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=seed)
    out.to_csv(output_path, index=False)
    print("Synthetic demo CSV saved:", output_path, out.shape)
    return str(output_path)


def infer_label_col(df, preferred=None):
    if preferred and preferred in df.columns:
        return preferred
    for col in LABEL_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(f"Could not infer label column. First columns: {list(df.columns)[:30]}")


def get_category(label):
    value = str(label).strip()
    if not value or value.lower() in {"nan", "none", "null"}:
        return "Unknown"
    if value in ATTACK_CATEGORY_MAP:
        return ATTACK_CATEGORY_MAP[value]

    lower_map = {key.lower(): val for key, val in ATTACK_CATEGORY_MAP.items()}
    lower_value = value.lower().strip()
    if lower_value in lower_map:
        return lower_map[lower_value]
    compact = re.sub(r"[^a-z0-9]+", "", lower_value)
    if compact in {"benign", "benigntraffic", "normal", "background"}:
        return "Benign"
    if "benign" in lower_value or lower_value == "0":
        return "Benign"

    prefix = re.split(r"[-_\s\.]+", lower_value, maxsplit=1)[0]
    prefix_map = {
        "ddos": "DDoS",
        "dos": "DoS",
        "mirai": "Mirai",
        "recon": "Recon",
        "scan": "Recon",
        "spoofing": "Spoofing",
        "mitm": "Spoofing",
        "web": "Web",
        "sqlinjection": "Web",
        "xss": "Web",
        "bruteforce": "BruteForce",
        "dictionarybruteforce": "BruteForce",
        "malicious": "Attack",
        "malware": "Attack",
        "attack": "Attack",
        "anomaly": "Attack",
    }
    return prefix_map.get(prefix, value)


def prepare_dataframe(df, label_col=None, dataset_name="dataset"):
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    label_col = infer_label_col(df, label_col)
    df[label_col] = df[label_col].astype(str).str.strip()
    df["Category"] = df[label_col].apply(get_category)
    df["BinaryLabel"] = (df["Category"] != "Benign").astype(int)

    drop_cols = []
    for col in df.columns:
        if col in {label_col, "Category", "BinaryLabel"}:
            continue
        col_lower = col.lower().strip()
        if any(re.search(pattern, col_lower) for pattern in LEAKAGE_PATTERNS):
            drop_cols.append(col)
    if drop_cols:
        print(
            f"[{dataset_name}] Dropping leakage/id-like columns:",
            drop_cols[:25],
            "..." if len(drop_cols) > 25 else "",
        )
        df = df.drop(columns=drop_cols, errors="ignore")

    for col in df.columns:
        if col in {label_col, "Category"}:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    feature_cols = [col for col in df.columns if col not in {label_col, "Category", "BinaryLabel"}]
    all_nan = [col for col in feature_cols if df[col].isna().all()]
    if all_nan:
        print(
            f"[{dataset_name}] Dropping all-NaN/non-numeric columns:",
            all_nan[:20],
            "..." if len(all_nan) > 20 else "",
        )
        df = df.drop(columns=all_nan)

    protected = {label_col, "Category", "BinaryLabel"}
    for col in [col for col in df.columns if col not in protected]:
        df[col] = (
            pd.to_numeric(df[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .astype("float32")
        )

    print(f"[{dataset_name}] shape={df.shape}, label_col={label_col}")
    print(f"[{dataset_name}] Binary counts:", df["BinaryLabel"].value_counts(dropna=False).to_dict())
    print(f"[{dataset_name}] Category counts:", df["Category"].value_counts(dropna=False).head(15).to_dict())
    return df.reset_index(drop=True), label_col


def load_prepared_csv(path, label_col=None, sample_rows=None, seed=42, dataset_name="dataset"):
    df = read_csv_maybe_sample(path, sample_rows=sample_rows, seed=seed)
    return prepare_dataframe(df, label_col=label_col, dataset_name=dataset_name)


def split_70_15_15(y, seed):
    y = np.asarray(y)
    idx = np.arange(len(y))
    first_split = StratifiedShuffleSplit(n_splits=1, train_size=0.70, random_state=seed)
    train_idx, temp_idx = next(first_split.split(idx, y))
    y_temp = y[temp_idx]
    second_split = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 99)
    val_rel, test_rel = next(second_split.split(temp_idx, y_temp))
    return train_idx, temp_idx[val_rel], temp_idx[test_rel]


def split_train_val(y, seed, val_size=0.15):
    idx = np.arange(len(y))
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    return next(splitter.split(idx, y))


def numeric_feature_columns(df, label_col):
    return [
        col
        for col in df.columns
        if col not in {label_col, "Category", "BinaryLabel"}
        and pd.api.types.is_numeric_dtype(df[col])
        and not df[col].isna().all()
    ]


def fit_transform_features(df, label_col, task, train_idx, val_idx, test_idx, feature_cols=None):
    feature_cols = feature_cols or numeric_feature_columns(df, label_col)
    X_train_df = df.iloc[train_idx][feature_cols].copy()
    X_val_df = df.iloc[val_idx][feature_cols].copy()
    X_test_df = df.iloc[test_idx][feature_cols].copy()

    medians = X_train_df.median(numeric_only=True).fillna(0.0)
    X_train_df = X_train_df.fillna(medians).fillna(0.0)
    X_val_df = X_val_df.fillna(medians).fillna(0.0)
    X_test_df = X_test_df.fillna(medians).fillna(0.0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df).astype(np.float32, copy=False)
    X_val = scaler.transform(X_val_df).astype(np.float32, copy=False)
    X_test = scaler.transform(X_test_df).astype(np.float32, copy=False)

    if task == "binary":
        y_train = df.iloc[train_idx]["BinaryLabel"].to_numpy(dtype=int)
        y_val = df.iloc[val_idx]["BinaryLabel"].to_numpy(dtype=int)
        y_test = df.iloc[test_idx]["BinaryLabel"].to_numpy(dtype=int)
        encoder = None
    else:
        encoder = LabelEncoder()
        y_train = encoder.fit_transform(df.iloc[train_idx]["Category"].astype(str).to_numpy())
        y_val = encoder.transform(df.iloc[val_idx]["Category"].astype(str).to_numpy())
        y_test = encoder.transform(df.iloc[test_idx]["Category"].astype(str).to_numpy())
    return X_train, y_train, X_val, y_val, X_test, y_test, encoder, scaler, medians, feature_cols


def transform_external_features(df, label_col, feature_cols, scaler, medians):
    Xdf = df.copy()
    for col in feature_cols:
        if col not in Xdf.columns:
            Xdf[col] = np.nan
    Xdf = Xdf[feature_cols].copy()
    for col in feature_cols:
        Xdf[col] = pd.to_numeric(Xdf[col], errors="coerce")
    Xdf = Xdf.replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0)
    return scaler.transform(Xdf).astype(np.float32, copy=False), df["BinaryLabel"].to_numpy(dtype=int)


def normalize_edge_columns(df):
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(".", "_", regex=False)
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    while any("__" in col for col in df.columns):
        df.columns = [col.replace("__", "_") for col in df.columns]
    return df


def prepare_edgeiiot_dataframe(
    df,
    dataset_name="Edge-IIoTset",
    drop_duplicates=True,
    use_author_style_drops=True,
):
    df = normalize_edge_columns(df)
    if "attack_label" not in df.columns and "attack_type" not in df.columns:
        raise ValueError("Edge-IIoTset must contain attack_label and/or attack_type columns.")

    if "attack_label" in df.columns:
        binary_s = pd.to_numeric(df["attack_label"], errors="coerce")
    else:
        binary_s = (~df["attack_type"].astype(str).str.lower().eq("normal")).astype(int)

    if "attack_type" in df.columns:
        category_s = df["attack_type"].astype(str).str.strip()
    else:
        category_s = np.where(binary_s.fillna(0).astype(int) == 0, "Normal", "Attack")

    category_s = pd.Series(category_s, index=df.index).replace({"Normal": "Benign", "normal": "Benign"})
    binary_out = binary_s.fillna((category_s != "Benign").astype(int)).astype(int)
    category_out = category_s.astype(str)

    drop_cols = ["attack_label", "attack_type", "mqtt_conack_flags", "mqtt_protoname"]
    if use_author_style_drops:
        drop_cols += EDGE_LEAKAGE_DROP_COLUMNS
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")

    label_like_cols = [
        col
        for col in df.columns
        if col.lower()
        in {
            "attack_label",
            "attack_type",
            "label",
            "binarylabel",
            "binary_label",
            "edgelabel",
            "edge_label",
            "edgecategory",
            "edge_category",
            "category",
        }
    ]
    if label_like_cols:
        print(f"[{dataset_name}] Dropping label-like aliases:", label_like_cols)
        df = df.drop(columns=label_like_cols, errors="ignore")

    remaining = [col for col in df.columns if "label" in col.lower() or "attack" in col.lower() or "category" in col.lower()]
    if remaining:
        print(f"[{dataset_name}] WARNING: Remaining suspicious columns:", remaining)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    all_nan = [col for col in df.columns if df[col].isna().all()]
    if all_nan:
        print(f"[{dataset_name}] Dropping all-NaN/non-numeric columns ({len(all_nan)}):", all_nan[:30])
        df = df.drop(columns=all_nan)

    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates(keep="first")
        after = len(df)
        print(f"[{dataset_name}] Dropped duplicates: {before - after:,} rows removed; remaining={after:,}")

    df[EDGE_LABEL_COL] = category_out.loc[df.index]
    df["Category"] = category_out.loc[df.index]
    df["BinaryLabel"] = binary_out.loc[df.index]

    protected = {EDGE_LABEL_COL, "Category", "BinaryLabel"}
    for col in [col for col in df.columns if col not in protected]:
        df[col] = (
            pd.to_numeric(df[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .astype("float32")
        )

    print(f"[{dataset_name}] shape={df.shape}, label_col={EDGE_LABEL_COL}")
    print(f"[{dataset_name}] Binary counts:", df["BinaryLabel"].value_counts(dropna=False).to_dict())
    print(f"[{dataset_name}] Category counts:", df["Category"].value_counts(dropna=False).head(20).to_dict())
    return df.reset_index(drop=True), EDGE_LABEL_COL


def load_edgeiiot_dataset(
    path,
    sample_rows=None,
    seed=42,
    drop_duplicates=True,
    use_author_style_drops=True,
):
    raw = read_csv_maybe_sample(path, sample_rows=sample_rows, seed=seed)
    return prepare_edgeiiot_dataframe(
        raw,
        dataset_name="Edge-IIoTset",
        drop_duplicates=drop_duplicates,
        use_author_style_drops=use_author_style_drops,
    )
