import pandas as pd

# load data
data = pd.read_csv("../Inputs/ball-by-ball prediction/main.csv", low_memory=False)

# Categorize features
match_features = data.columns[:9].to_list()
batsman_history = data.columns[9:18].to_list()
bowler_history = data.columns[18:27].to_list()
data.columns = (
    match_features
    + [f"hist_{col}" for col in batsman_history]
    + [f"hist_{col}" for col in bowler_history]
    + ["target"]
)
df = pd.DataFrame()
df["target"] = data["target"]
to_text = lambda row: " ".join([f"{k}: {v}," for k, v in row.items()])
df["features"] = data.iloc[:, :-1].apply(to_text, axis=1)

# print the list of text strings
df.to_csv("../Inputs/ball-by-ball prediction/main_text.csv", index=False)
