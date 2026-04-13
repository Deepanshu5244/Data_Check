import requests
import pandas as pd
import json

# =========================
# CONFIG
# =========================
TAGS = [
    'MAL_TBE.U40HNA11CQ902_XJ60',
    'MAL_TBE.U40LBA11FF901_XJ60',
]

TAGMETA_URL = "https://data.exactspace.co/exactapi/units/6863ac729154090006e03b3b/tagmeta"
KAIROS_API = "https://data.exactspace.co/api/v1/datapoints/query"

ACCESS_TOKEN = "SQFkEv3y4rbPYicpANnr0QlohH49NHUKxGmnVigzPMPD7XthRNJcC9wnwDi4MUjz"

HEADERS = {
    "Authorization": ACCESS_TOKEN,
    "Content-Type": "application/json"
}

START_TIME = 1743058475000
END_TIME = 1776086795000


# =========================
# STEP 1: FETCH TAG META
# =========================
def get_equipment_states(tags):
    eqp_set = set()
    tag_info = []

    for tag in tags:
        params = {
        "filter": json.dumps({
            "where": {"dataTagId": tag},
            "fields": ["equipmentId", "description"]
        }),
    "access_token": ACCESS_TOKEN
}

        try:
            data = requests.get(TAGMETA_URL, params=params).json()

            if data and isinstance(data, list) and len(data) > 0:
                eqp = data[0].get("equipmentId")
                desc = data[0].get("description")

                if eqp:
                    eqp_set.add("state__" + eqp)

                tag_info.append({
                    "tag": tag,
                    "equipmentId": eqp,
                    "description": desc
                })

                print(f"tag = {tag:<40} → equipmentId = {eqp} → description = {desc}")

            else:
                print(f"tag = {tag:<40} → No data")

        except Exception as e:
            print(f"tag = {tag:<40} → Failed ({e})")

    return list(eqp_set), tag_info


# =========================
# STEP 2: BUILD PAYLOAD
# =========================
def build_payload(tags, start, end):
    return {
        "metrics": [{"name": str(tag)} for tag in tags],
        "plugins": [],
        "cache_time": 0,
        "start_absolute": start,
        "end_absolute": end
    }


# =========================
# STEP 3: FORMAT RESPONSE
# =========================
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
            df = df.sort_values("time")

            if final_df is None:
                final_df = df
            else:
                final_df = pd.merge_asof(
                    final_df.sort_values("time"),
                    df.sort_values("time"),
                    on="time",
                    tolerance=60000,
                    direction="nearest"
                )

        except Exception as e:
            print(f"Error processing tag: {e}")

    return final_df if final_df is not None else pd.DataFrame()


# =========================
# STEP 4: FETCH DATA
# =========================
def fetch_data(tags, start, end):
    payload = build_payload(tags, start, end)

    try:
        response = requests.post(KAIROS_API, json=payload, headers=HEADERS)

        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code}")
            return pd.DataFrame()

        resultset = response.json()

    except Exception as e:
        print(f"❌ Request failed: {e}")
        return pd.DataFrame()

    df = format_result_to_df(resultset)

    if not df.empty:
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df.set_index("time", inplace=True)

    return df


# =========================
# STEP 5: FILTER ACTIVE STATES
# =========================
def filter_active_states(df):
    if df.empty:
        return df

    df = df.dropna()
    state_cols = [col for col in df.columns if col.startswith("state__")]

    if not state_cols:
        return df

    return df[(df[state_cols] == 1).all(axis=1)]


# =========================
# MAIN
# =========================
def main():
    print("🚀 Starting Pipeline...\n")

    print("1️⃣ Fetching equipment states...")
    eqp_states, tag_info = get_equipment_states(TAGS)
    print(f"   ✅ Found {len(eqp_states)} state tags\n")

    all_tags = TAGS + eqp_states

    print("2️⃣ Fetching time-series data...")
    df = fetch_data(all_tags, START_TIME, END_TIME)
    print(f"   📊 Raw shape: {df.shape}\n")

    print("3️⃣ Filtering active states...")
    final_df = filter_active_states(df)
    print(f"   ✅ Filtered shape: {final_df.shape}\n")

    print("🎉 Pipeline Completed!")
    print(final_df.head())

    return final_df, tag_info


# =========================
# RUN
# =========================



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

# if __name__ == "__main__":
#     final_df, tag_info = main()

#     print("\nFinal Shape:", final_df.shape)
#     print(final_df.head())
