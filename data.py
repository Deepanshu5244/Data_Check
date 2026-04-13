import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# =========================
# CONFIG
# =========================
BASE_URL = "https://bpcl.exactspace.co"
UNITS_ID = "682db43920f43b00077a5d52"
MODEL_API = f"{BASE_URL}/exactapi/modelpipelines"
TAGMETA_API = f"{BASE_URL}/exactapi/units/{UNITS_ID}/tagmeta"
KAIROS_API = f"{BASE_URL}/api/v1/datapoints/query"

AUTH_TOKEN = "rIy8DDzEiWCjbFHZueLpHCzh84PLSdUTgPP3DLG5oGZ5a4M2jmuiLxXymWFFML1Z"

TARGET_OUTPUT = "BPMR1_SRUSPI1851C.PV"

START_TIME = 1744109508000
END_TIME =  1775645508000

HEADERS = {
    "Authorization": AUTH_TOKEN,
    "Content-Type": "application/json"
}

# =========================
# Existing Functions (same as before)
# =========================
def get_tags_from_model():
    filter_query = {"where": {"unitsId": UNITS_ID}, "fields": {"deployVersion": True, "performance": True}}
    params = {"filter": json.dumps(filter_query)}
    response = requests.get(MODEL_API, headers=HEADERS, params=params)
    data = response.json()

    results = []
    for item in data:
        dv = item.get("deployVersion")
        if dv and dv.startswith("V"):
            try:
                idx = int(dv[1:]) - 1
                perf = item.get("performance", [])[idx]
                if perf.get("outputTag") == TARGET_OUTPUT:
                    results.append(perf.get("outputTag"))
                    results.extend(perf.get("selectedVars", []))
            except:
                continue
    return list(set(results))


def get_equipment_states(tags):
    eqp_set = set()
    tag_info = []
    for tag in tags:
        params = {"filter": f'{{"where":{{"dataTagId":"{tag}"}},"fields":["equipmentId","description"]}}'}
        try:
            data = requests.get(TAGMETA_API, params=params, headers=HEADERS).json()
            print("*"*30)
            print(data)
            if data and isinstance(data, list):
                eqp = data[0].get("equipmentId")
                desc=data[0].get("description")
                tag_info.append({
                "tag": tag,
                "equipmentId": eqp,
                "description": desc
            })
                print(f"Tag: {tag} → Equipment ID: {eqp} → Description: {desc}")
                if eqp:
                    eqp_set.add("state__" + eqp)
        except Exception as e:
            print(f"Error fetching equipment for tag {tag}: {e}")
            
    return list(eqp_set), tag_info


def build_payload(tags, start, end):
    return {
        "metrics": [{"name": str(tag)} for tag in tags],
        "plugins": [],
        "cache_time": 0,
        "start_absolute": start,
        "end_absolute": end
    }


def format_result_to_df(resultset):
    if not resultset or "queries" not in resultset:
        return pd.DataFrame()
    final_df = None
    for res in resultset["queries"]:
        try:
            values = res["results"][0]["values"]
            tag_name = res["results"][0]["name"]
            if not values:
                continue
            df = pd.DataFrame(values, columns=["time", tag_name])
            if final_df is None:
                final_df = df
            else:
                final_df = pd.merge_asof(final_df.sort_values("time"), df.sort_values("time"),
                                         on="time", tolerance=60000, direction="nearest")
        except:
            continue
    return final_df if final_df is not None else pd.DataFrame()


def fetch_timeseries(tags, start, end):
    payload = build_payload(tags, start, end)
    try:
        res = requests.post(KAIROS_API, json=payload, headers=HEADERS)
        if res.status_code != 200:
            print("❌ API Error:", res.status_code)
            return pd.DataFrame()
        df = format_result_to_df(res.json())
        if not df.empty:
            df["time"] = pd.to_datetime(df["time"], unit="ms")
            df.set_index("time", inplace=True)
        return df
    except Exception as e:
        print("❌ Fetch failed:", e)
        return pd.DataFrame()


def filter_active_states(df):
    if df.empty:
        return df
    df = df.dropna()
    state_cols = [c for c in df.columns if c.startswith("state__")]
    if not state_cols:
        return df
    return df[(df[state_cols] == 1).all(axis=1)]


# =========================
# PLOTTING FUNCTION FOR VS CODE
# =========================
def plot_before_after(raw_df, filtered_df, tag_desc_map, filename=TARGET_OUTPUT):
    if raw_df.empty:
        print("⚠️ No data to plot")
        return

    plot_cols = [col for col in raw_df.columns if not col.startswith("state__")]

    n = len(plot_cols)
    fig, axes = plt.subplots(n, 1, figsize=(14, 5 * n), sharex=True)

    if n == 1:
        axes = [axes]

    for i, col in enumerate(plot_cols):
        ax = axes[i]

        if col in raw_df.columns:
            ax.plot(raw_df.index, raw_df[col], label='Before (Raw)', alpha=0.8)

        if col in filtered_df.columns and not filtered_df.empty:
            ax.plot(filtered_df.index, filtered_df[col], label='After (Cleaned)', linewidth=1.8)

        desc = tag_desc_map.get(col, "No Description")

        ax.set_title(f"Variable: {col} | {desc}", fontsize=13)
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    plt.xlabel("Time")
    plt.tight_layout()

    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"✅ Plot saved as: {filename}")

# =========================
# MAIN
# =========================
def main():
    print("🚀 Starting Pipeline...\n")

    print("1️⃣ Fetching tags from model...")
    tags = get_tags_from_model()
    print(f"   ✅ Found {tags} tags\n")

    print("2️⃣ Fetching equipment states...")
    eqp_states, tag_info = get_equipment_states(tags)
    print(f"   ✅ Found {len(eqp_states)} equipment state tags\n")
    tag_desc_map = {item["tag"]: item["description"] for item in tag_info}
    all_tags = tags + eqp_states

    print("3️⃣ Fetching time-series data (this may take time)...")
    raw_df = fetch_timeseries(all_tags, START_TIME, END_TIME)
    print(f"   📊 Raw data shape: {raw_df.shape}\n")

    print("4️⃣ Filtering active states...")
    filtered_df = filter_active_states(raw_df)
    print(f"   ✅ Filtered shape: {filtered_df.shape}\n")

    # Plot Before vs After
    print("5️⃣ Generating Before vs After plot...")
    plot_before_after(raw_df, filtered_df, tag_desc_map, filename=f"{TARGET_OUTPUT}.png")

    # Final clean dataframe
    final_df = filtered_df[[col for col in filtered_df.columns if not col.startswith("state__")]]

    print("\n🎉 Pipeline Completed!")
    print(f"Final Clean Data Shape: {final_df.shape}")
    print("\nFirst 5 rows:")
    print(final_df.head())

    # Optional: Save CSV
    # final_df.to_csv("final_clean_data.csv")
    # print("Data also saved as final_clean_data.csv")

    return final_df,tag_info


def get_final_data():
    final_df, tag_info = main()

    # Drop state columns
    final_df = final_df[[col for col in final_df.columns if not col.startswith("state__")]]

    return final_df, tag_info


# =========================
# RUN
# =========================


if __name__ == "__main__":
    final_df = get_final_data()

    print("\n🔍 Final Clean Output Shape:", final_df.shape)
    print(final_df.head())