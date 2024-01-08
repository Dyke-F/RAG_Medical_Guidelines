# create a visualisation of vectordatabase
from dotenv import load_dotenv
import chromadb
import numpy as np
from pathlib import Path
from collections import defaultdict
from langchain.embeddings import OpenAIEmbeddings
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from matplotlib.colors import BoundaryNorm
from types import SimpleNamespace

load_dotenv()

# Langchain-Chroma natively does not appear to return embeddings instead of documents, so extend the Chroma API to do so
from langchain.vectorstores import Chroma
def extend_chroma(cls):
    def raw_similarity_search_with_score(self, query, k=4, filter=None, **kwargs):
        if self._embedding_function is None:
            # access the private attribute ...
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


def get_all_index_paths(base_path):
    return [path for subpath in Path(base_path).iterdir() if subpath.is_dir() for path in subpath.iterdir() if path.is_dir()]


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


def concat_sources(embd_store):
    cat = defaultdict(list)
    lens = {}
    for source in embd_store.keys():
        for key, value in embd_store[source][0].items():
            cat[key].extend(value)
            lens[source] = len(value)
    
    # should work because dict iteration presevers key insertion order
    for source in embd_store.keys():
        cat["source"] += [source] * lens[source] 
    cat["embeddings"] = np.array(cat["embeddings"])
    return cat


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



path = "/mnt/bulk/dferber/catchy/LLM_Vector/Vector_LLM_for_Medical_QA/chroma_db"
index_paths = get_all_index_paths(path)

def main():
    index_paths = get_all_index_paths(path)
    embd_store = {}
    for idx_path in index_paths:
        embd_store[f"{idx_path.parent.name}_{idx_path.name}"] = load_embeddings(idx_path)
    complete = concat_sources(embd_store)
    data_w_transforms = transform_tsne(complete)
    visualize_tsne(data_w_transforms)
