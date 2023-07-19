import os

import openai
import pandas as pd
import numpy as np

datafile_path = "fine_food_reviews_with_embeddings_1k.csv"

df = pd.read_csv(datafile_path)
df["embedding"] = df.embedding.apply(eval).apply(np.array)

# setting api key
os.environ['OPENAI_API_KEY'] = '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'
openai.api_key = os.getenv("OPENAI_API_KEY")
# models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"
from openai.embeddings_utils import get_embedding, cosine_similarity


# search through the reviews for a specific product
def search_reviews(df, product_description, n=3, pprint=True):
    product_embedding = get_embedding(
        product_description,
        engine=EMBEDDING_MODEL
    )
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, product_embedding))

    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
        .combined.str.replace("Title: ", "")
        .str.replace("; Content:", ": ")
    )
    if pprint:
        for r in results:
            print(r[:1000])
            print()
    
    return results


search_item = input("Enter a item you want review for:")
results = search_reviews(df, search_item, n=3)
