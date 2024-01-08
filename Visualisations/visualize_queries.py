from dotenv import load_dotenv
import chromadb
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from langchain.embeddings import OpenAIEmbeddings
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from matplotlib.colors import BoundaryNorm
from types import SimpleNamespace
from collections import Counter

load_dotenv()

from langchain.vectorstores import Chroma
def extend_chroma(cls):
    def raw_similarity_search_with_score(self, query, k=4, filter=None, **kwargs):
        if self._embedding_function is None:
            # accessing private attribute ...
            results = self._Chroma__query_collection(
                query_texts=[query], n_results=k, where=filter
            )
        else:
            query_embedding = self._embedding_function.embed_query(query)
            results = self._Chroma__query_collection(
                query_embeddings=[query_embedding], n_results=k, where=filter
            )
        return results
    
    setattr(cls, 'raw_similarity_search_with_score', raw_similarity_search_with_score)
    return cls

Chroma = extend_chroma(Chroma)


def get_all_index_paths(base_path, entity):
    return [path for subpath in Path(base_path).iterdir() if subpath.is_dir() for path in subpath.iterdir() if path.is_dir() and entity in str(path)]


def load_embeddings(index_path):
    embeddings = OpenAIEmbeddings()
    client_settings = chromadb.config.Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=str(index_path),
        anonymized_telemetry=False
    )
    vectorstore = Chroma(persist_directory=index_path, embedding_function=embeddings, client_settings=client_settings)
    data = vectorstore._collection.get(include=["embeddings", "documents", "metadatas"])
    return SimpleNamespace(data=data, vectorstore=vectorstore)


def get_ids_for_query(embd_store, queries):
    query_to_ids = {}
    for query in queries:
        ids = []
        for key, data_vs in embd_store.items():
            vs_store = getattr(data_vs, "vectorstore")
            embeds = vs_store.raw_similarity_search_with_score(query, k=15)["ids"][0]
            ids.extend(embeds)
        query_to_ids[query] = ids

    all_ids = [id for sublist in query_to_ids.values() for id in sublist]
    item_counts = Counter(all_ids)
    multiple = [item for item, count in item_counts.items() if count > 1]

    # remove ids from the original list if appear more than once
    for key, value in query_to_ids.items():
        query_to_ids[key] = [item for item in value if item not in multiple]
    query_to_ids["multiple"] = multiple
    
    return query_to_ids


def concat_sources(embd_store):
    cat = defaultdict(list)
    lens = {}
    for source, value in embd_store.items():
        data = value.data
        for k, v in data.items():
            cat[k].extend(v)
            lens[source] = len(v)
            
    # should work because dict iteration presevers key insertion order
    for source in embd_store.keys():
        cat["source"] += [source] * lens[source] 
    # cat["embeddings"] = np.array(cat["embeddings"])
    return cat


def merge_to_df(query_to_ids, concat):
    result = []
    for key, values in query_to_ids.items():
        result.extend([(key, value) for value in values])

    query_df = pd.DataFrame(result, columns=['query', 'query_ids'])
    concat_df = pd.DataFrame(concat)

    merged = pd.merge(concat_df, query_df, left_on="ids", right_on="query_ids", how="left")
    merged.drop("query_ids", axis=1, inplace=True)
    merged.fillna("unused", inplace=True)

    return merged


def transform_tsne(data, scale_features=True, n_components=2, init="pca"):
    embeddings = data["embeddings"]
    np_data = np.array(embeddings)
    if scale_features:
        np_data = StandardScaler().fit_transform(np_data)
    tsne = TSNE(
        n_components=n_components,
        init=init,
        perplexity=30, #early_exaggeration=12, learning_rate=200,   
        random_state=123
        )
    data_transformed = tsne.fit_transform(np_data)
    data["data_transformed"] = data_transformed
    return data


def visualize_tsne(data):
    data_transformed = data["data_transformed"]
    labels = data["source"]
    lab_enc = LabelEncoder()
    labels_encoded = lab_enc.fit_transform(labels)

    fig, ax = plt.subplots(figsize=(5,4))
    cmap = plt.get_cmap('viridis', len(lab_enc.classes_))
    norm = BoundaryNorm(range(len(lab_enc.classes_)+1), cmap.N)

    scatter = ax.scatter(data_transformed[:, 0], data_transformed[:, 1], c=labels_encoded, cmap=cmap, norm=norm, alpha=0.6)
    cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.05])  # [left, bottom, width, height]
    cbar = plt.colorbar(scatter, cax=cbar_ax, ticks=np.arange(len(lab_enc.classes_)))
    midpoints = np.arange(len(lab_enc.classes_)) + 0.5
    cbar.set_ticks(midpoints)
    cbar.set_ticklabels(lab_enc.classes_)
    cbar.set_label('Target Labels')
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    ax.set_title('t-SNE Visualization', fontweight='bold')

    plt.show()