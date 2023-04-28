# A python script to create openAI embeddings for the given dataset of format (text, label)
# Usage: python create_embeddings.py <start_index> <end_index>
# Example: python create_embeddings.py 0 10000
# This will create embeddings for the first 10000 rows of the dataset


import argparse
import pandas as pd
import openai
import os
import dotenv
import numpy as np
import time

config = dotenv.dotenv_values(".env")
openai.api_key = config["OPENAI_API_KEY"]


def get_embedding(text, model="text-embedding-ada-002"):
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]


def emb2feats(embs):
    emb_names = [f"embfeat_{i}" for i in range(1, 1537)]
    df = pd.DataFrame(
        embs.ada_embedding.apply(eval).apply(np.array).tolist(), columns=emb_names
    )
    df["target"] = embs.target
    return df


# parser = argparse.ArgumentParser()
# parser.add_argument("start_index", type=int, help="start index of the data")
# parser.add_argument("end_index", type=int, help="end index of the data")
# # parser.add_argument("--model_name", type=str, required=True)
# args = parser.parse_args()

data = pd.read_csv("../Inputs/ball-by-ball prediction/embeddings_0.csv")
# df = data[args.start_index : args.end_index].copy()
# print(
#     f"start_index: {args.start_index}, end_index: {args.end_index}, total: {len(data)}"
# )
# print(
#     "Approx. cost to compute embeddings (in $): ",
#     (len(df) * len(df.features[0]) / (4 * 1000)) * 0.0004,
# )
# print(df)

# # time this block
# start = time.time()
# df["ada_embedding"] = df.features.apply(
#     lambda x: get_embedding(x, model="text-embedding-ada-002")
# )
# end = time.time()
# print(f"Time taken: {end-start} seconds")

emb2feats(data).to_csv("../Inputs/ball-by-ball prediction/embfeats10K.csv", index=False)

# df.to_csv(
#     f"../Inputs/ball-by-ball prediction/embeddings_{args.start_index}.csv", index=False
# )

# 10000 20000
# 20000 30000
# 30000 40000
# 40000 50000
# 50000 60000
# 60000 70000
# 70000 79747
