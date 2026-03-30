"""
Step 2: Download the NSL-KDD dataset.

NSL-KDD is the most commonly used benchmark dataset for network intrusion
detection research. It is an improved version of the original KDD Cup 1999
dataset, removing duplicate records and balancing difficulty levels.

Usage:
    python -m src.download_data
"""

import os
import urllib.request
import zipfile


# NSL-KDD dataset download URL (University of New Brunswick mirror)
DATA_URL = "https://web.archive.org/web/2024/https://iscxdownloads.cs.unb.ca/iscxdownloads/NSL-KDD/NSL-KDD.zip"
# Alternative: use the individual CSV files from a reliable mirror
TRAIN_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
TEST_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"

RAW_DIR = os.path.join("data", "raw")

# The 41 features in the NSL-KDD dataset + label + difficulty
COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty_level"
]

# Attack type to category mapping
ATTACK_CATEGORY = {
    "normal": "normal",
    # DoS attacks
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "mailbomb": "DoS",
    "apache2": "DoS", "processtable": "DoS", "udpstorm": "DoS",
    # Probe attacks
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "satan": "Probe", "mscan": "Probe", "saint": "Probe",
    # R2L attacks
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L",
    "warezclient": "R2L", "warezmaster": "R2L", "snmpgetattack": "R2L",
    "named": "R2L", "xlock": "R2L", "xsnoop": "R2L",
    "sendmail": "R2L", "httptunnel": "R2L", "worm": "R2L",
    "snmpguess": "R2L",
    # U2R attacks
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "xterm": "U2R", "ps": "U2R",
    "sqlattack": "U2R",
}


def download_file(url, filepath):
    """Download a file from URL to the given filepath."""
    if os.path.exists(filepath):
        print(f"  [skip] {filepath} already exists")
        return
    print(f"  Downloading {url} ...")
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"  [done] Saved to {filepath}")
    except Exception as e:
        print(f"  [error] Failed to download: {e}")
        print(f"  Please manually download the NSL-KDD dataset and place it in {RAW_DIR}/")
        raise


def main():
    """Download NSL-KDD dataset files."""
    os.makedirs(RAW_DIR, exist_ok=True)

    print("=" * 60)
    print("Downloading NSL-KDD Dataset")
    print("=" * 60)

    train_path = os.path.join(RAW_DIR, "KDDTrain+.txt")
    test_path = os.path.join(RAW_DIR, "KDDTest+.txt")

    download_file(TRAIN_URL, train_path)
    download_file(TEST_URL, test_path)

    # Save column names for reference
    col_path = os.path.join(RAW_DIR, "columns.txt")
    with open(col_path, "w") as f:
        f.write("\n".join(COLUMN_NAMES))
    print(f"  [done] Column names saved to {col_path}")

    print("\nDataset download complete!")
    print(f"  Training set: {train_path}")
    print(f"  Test set:     {test_path}")
    print(f"  Total columns: {len(COLUMN_NAMES)}")


if __name__ == "__main__":
    main()
