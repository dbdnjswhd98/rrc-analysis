import pandas as pd
from pathlib import Path

# =========================
# 0. 디바이스별 UE IP 설정
# =========================
UE_IPS = {
    "S22":   "10.215.173.1",
    "S9+":   "10.215.173.1",
    "PocoF1":"10.215.173.1",
}

BASE_DIR = Path(__file__).parent.parent
UE_PCAP_DIR = BASE_DIR / "data" / "raw" / "ue_pcap"
ENB_S1AP_DIR = BASE_DIR / "data" / "raw" / "enb_s1ap"
OUT_DIR = BASE_DIR / "data" / "processed" / "processed_testbed"
OUT_DIR.mkdir(exist_ok=True, parents=True)


# =========================
# 1. S1AP → 1초 단위 RRC
# =========================
def build_rrc_timeline_from_s1ap(s1ap_csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(s1ap_csv_path)
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values('Time')

    t_start = df['Time'].min().floor('S')
    t_end   = df['Time'].max().floor('S')

    intervals = []
    current_start = None

    for _, row in df.iterrows():
        info = str(row['Info'])
        t = row['Time']

        # RRC Connected 시작 추정
        if current_start is None and (
            "InitialUEMessage" in info
            or "Attach request" in info
            or "Service request" in info
        ):
            current_start = t

        # RRC Connected 종료 추정
        if current_start is not None and (
            "UEContextReleaseComplete" in info
            or "UEContextReleaseCommand" in info
        ):
            intervals.append((current_start, t))
            current_start = None

    if current_start is not None:
        intervals.append((current_start, t_end))

    time_index = pd.date_range(t_start, t_end, freq="1S")
    rrc = pd.DataFrame({"Time_1s": time_index})
    rrc["rrc_state"] = 0

    for s, e in intervals:
        mask = (rrc["Time_1s"] >= s.floor("S")) & (rrc["Time_1s"] <= e.floor("S"))
        rrc.loc[mask, "rrc_state"] = 1

    return rrc


# =========================
# 2. UE pcap → 1초 단위 트래픽
# =========================
def aggregate_pcap_1s(pcap_csv_path: Path, ue_ip: str) -> pd.DataFrame:
    df = pd.read_csv(pcap_csv_path)
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.sort_values("Time")

    t_start = df["Time"].min().floor("S")
    t_end   = df["Time"].max().floor("S")

    df["is_ul"] = df["Source"] == ue_ip
    df["is_dl"] = df["Destination"] == ue_ip

    def resample_dir(mask_col: str) -> pd.DataFrame:
        d = df[df[mask_col]].copy()
        if d.empty:
            idx = pd.date_range(t_start, t_end, freq="1S")
            return pd.DataFrame({"Time_1s": idx, "bytes": 0, "pkts": 0})

        d = d.set_index("Time")
        agg = d.resample("1S").agg({
            "Length": "sum",
            "No.": "count"
        }).rename(columns={"Length": "bytes", "No.": "pkts"})

        agg = agg.reindex(pd.date_range(t_start, t_end, freq="1S"), fill_value=0)
        agg.index.name = "Time_1s"
        return agg.reset_index()

    ul = resample_dir("is_ul")
    dl = resample_dir("is_dl")

    ts = ul.merge(dl, on="Time_1s", suffixes=("_ul", "_dl"))

    ts = ts.rename(columns={
        "bytes_ul": "ul_bytes",
        "pkts_ul":  "ul_pkts",
        "bytes_dl": "dl_bytes",
        "pkts_dl":  "dl_pkts",
    })

    return ts


# =========================
# 3. merge + 메타데이터
# =========================
def merge_rrc_pcap(rrc_ts: pd.DataFrame, pcap_ts: pd.DataFrame) -> pd.DataFrame:
    start_time = max(rrc_ts["Time_1s"].min(), pcap_ts["Time_1s"].min())
    end_time   = min(rrc_ts["Time_1s"].max(), pcap_ts["Time_1s"].max())

    r = rrc_ts[(rrc_ts["Time_1s"] >= start_time) & (rrc_ts["Time_1s"] <= end_time)]
    p = pcap_ts[(pcap_ts["Time_1s"] >= start_time) & (pcap_ts["Time_1s"] <= end_time)]

    merged = pd.merge(r, p, on="Time_1s", how="inner").sort_values("Time_1s")

    t0 = merged["Time_1s"].iloc[0]
    merged["time_sec"] = (merged["Time_1s"] - t0).dt.total_seconds().astype(int)

    return merged


def add_metadata(df: pd.DataFrame, device: str, env: str, wifi: str) -> pd.DataFrame:
    df["device"] = device            # "S22" / "S9+" / "PocoF1"
    df["env"]    = env               # "Clean" / "Dirty"
    df["wifi"]   = wifi              # "Off" / "On"
    df["network"] = "Testbed"
    return df


# =========================
# 4. 시나리오 정의 (12개)
# =========================
SCENARIOS = [
    # device, env, wifi, pcap 파일, s1ap 파일
    ("S22",   "Clean", "Off", "S22_test.csv",           "S22.csv"),
    ("S22",   "Dirty", "Off", "S22_test_apps.csv",      "S22_apps.csv"),
    ("S22",   "Clean", "On",  "S22_test_wifi.csv",      "S22_wifi.csv"),
    ("S22",   "Dirty", "On",  "S22_test_wifi_apps.csv", "S22_wifi_apps.csv"),

    ("S9+",   "Clean", "Off", "S9+_test.csv",           "S9+.csv"),
    ("S9+",   "Dirty", "Off", "S9+_test_apps.csv",      "S9+_apps.csv"),
    ("S9+",   "Clean", "On",  "S9+_test_wifi.csv",      "S9+_wifi.csv"),
    ("S9+",   "Dirty", "On",  "S9+_test_wifi_apps.csv", "S9+_wifi_apps.csv"),

    ("PocoF1","Clean", "Off", "PocoF1_test.csv",           "PocoF1.csv"),
    ("PocoF1","Dirty", "Off", "PocoF1_test_apps.csv",      "PocoF1_apps.csv"),
    ("PocoF1","Clean", "On",  "PocoF1_test_wifi.csv",      "PocoF1_wifi.csv"),
    ("PocoF1","Dirty", "On",  "PocoF1_test_wifi_apps.csv", "PocoF1_wifi_apps.csv"),
]


# =========================
# 5. 메인: 12개 한 번에 처리
# =========================
def process_all():
    for device, env, wifi, pcap_name, s1ap_name in SCENARIOS:
        print(f"[+] Processing {device}, {env}, WiFi {wifi} ...")

        ue_ip = UE_IPS[device]

        pcap_path = UE_PCAP_DIR / pcap_name
        s1ap_path = ENB_S1AP_DIR / s1ap_name

        rrc_ts = build_rrc_timeline_from_s1ap(s1ap_path)
        pcap_ts = aggregate_pcap_1s(pcap_path, ue_ip)
        merged = merge_rrc_pcap(rrc_ts, pcap_ts)
        merged = add_metadata(merged, device, env, wifi)

        out_name = f"{device}_{env}_WiFi{wifi}.csv"
        out_path = OUT_DIR / out_name
        merged.to_csv(out_path, index=False)

        print(f"    -> saved to {out_path}")

if __name__ == "__main__":
    process_all()
