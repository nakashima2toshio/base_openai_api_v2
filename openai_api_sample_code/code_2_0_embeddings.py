# embeddings from document and cook book
from openai import OpenAI
import pandas as pd
import tiktoken
import pprint

model_embedding_3_small = "text-embedding-3-small"
client = OpenAI()


def create_embeddings2(input_text, model_name, encoding_format):
    # 入力テキストを表す埋め込みベクトルを作成します。
    # (ex.) input="The food was delicious and the waiter..."
    response = client.embeddings.create(
        model=model_name,
        input=input_text,
        encoding_format="float"
    )
    return response

def create_embeddings(input_text, model_name):
    response = client.embeddings.create(
        input="Your text string goes here",
        model="text-embedding-3-small"
    )

    return response.data[0].embedding

"""
データセットから埋め込みを取得する:
https://cookbook.openai.com/examples/get_embeddings_from_dataset
"""
def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

df['ada_embedding'] = df.combined.apply(lambda x: get_embedding(x, model='text-embedding-3-small'))
df.to_csv('output/embedded_1k_reviews.csv', index=False)



def main():
    input_text = "The food was delicious and the waiter..."
    model_name = model_embedding_3_small
    res = create_embeddings(input_text, model_name)
    pprint.pprint(res)


if __name__ == "__main__":
    main()

