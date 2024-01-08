import asyncio
import json
import os
from pathlib import Path

import pandas as pd
from llama_index import ServiceContext
from llama_index.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.llms import OpenAI

from dotenv import load_dotenv
from getpass import getpass
import time
import fire
from tqdm import tqdm


# set JSON PATHs HERE
gpt4_rag_results = Path("./Results/GPT4_RAG/gpt-4_outputs.json")
naive_gpt4_results = Path("./Results/GPT4_noRAG").rglob("*.json")


def read_json_file(file_path):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        with open(file_path, "r") as f:
            content = f.read()
            # workaround because multiple json objects are stored as {}{}{} in the file
            data = json.loads(f'[{content.replace("}{", "},{")}]')
    return data


def read_naive_gpt4_json_file(file_paths):
    data = []
    for file_path in file_paths:
        with open(file_path, "r") as f:
            data.extend(eval(f.read()))
    return data


async def evaluate_retrieval_engine(query_to_source_docs, save_path, save_name):
    r"""Asynchronously evaluate a retrieval engine on a list of questions.
    Uses llama_indexes's FaithfulnessEvaluator and RelevancyEvaluator.
    """

    gpt4_eval = OpenAI(model="gpt-4-0613", temperature=0)
    gpt4_eval_service_context = ServiceContext.from_defaults(llm=gpt4_eval)
    faithfulness_evaluator = FaithfulnessEvaluator(
        service_context=gpt4_eval_service_context
    )
    relevancy_evaluator = RelevancyEvaluator(service_context=gpt4_eval_service_context)

    result_list = []
    for item in tqdm(query_to_source_docs):
        query = item["query"]
        response = item["response"]
        contexts = item["source_docs"]
        faithfulness = await faithfulness_evaluator.aevaluate(
            query=query, response=response, contexts=contexts
        )
        relevancy = await relevancy_evaluator.aevaluate(
            query=query, response=response, contexts=contexts
        )
        result_list.append(
            {
                "query": query,
                "response": response,
                "faithfulness": faithfulness.passing,
                "relevancy": relevancy.passing,
            }
        )
        time.sleep(60)

    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    df = pd.DataFrame(result_list)
    df.to_csv(save_path + "/" + save_name + ".csv")

    return df


def main(save_path, save_name, run_rag=True):
        load_dotenv()
        os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY") or getpass(
            f"Enter a valid OpenAI API key: "
        )
        assert os.environ["OPENAI_API_KEY"].startswith("sk-"), f"Invalid OpenAI API key"


        json_data = read_json_file(gpt4_rag_results)

        query_to_source_docs_with_RAG = []
        for item in json_data:
            item_source_chunks = []
            if "source documents" in item and isinstance(item["source documents"], list):
                for doc_list in item["source documents"]:
                    if isinstance(doc_list, list):
                        # first collection either ASCO or ESMO
                        for doc in doc_list[0]:
                            if "page_content" in doc:
                                item_source_chunks.append(doc["page_content"])
                        # second collection either ASCO or ESMO
                        for doc in doc_list[1]:
                            if "page_content" in doc:
                                item_source_chunks.append(doc["page_content"])

            query_to_source_docs_with_RAG.append(
                {
                    "query": item["Human Message"][0],
                    "response": item["AI Response"][0],
                    "source_docs": item_source_chunks,
                }
            )

        if run_rag:
            # if this folder does not exist run the evaluation on the GPT4 + RAG data
            df = asyncio.run(
                evaluate_retrieval_engine(query_to_source_docs_with_RAG, save_path)
            )
            
        else:
            # if it has been run already use the query_to_source_docs as a lookup table to parse the relevant inforamtion into the naive GPT-4 approach to check for potential hallucination
            naive_gpt4_data = read_naive_gpt4_json_file(naive_gpt4_results)
            
            map_with_without_RAG = {item["query"]: item["source_docs"] for item in query_to_source_docs_with_RAG}
            query_to_source_docs = [] # we match the gpt4 RAG source data to the naive gpt4 results
            for item in naive_gpt4_data:
                query = item["topic"]
                response = item["response"]
                source_docs = map_with_without_RAG[query]
                d = {"query": query, "response": response, "source_docs": source_docs}
                query_to_source_docs.append(d)

            df = asyncio.run(
                evaluate_retrieval_engine(query_to_source_docs, save_path, save_name)
            )

if __name__ == "__main__":
    main("RetrievalEvaluations", "faithfulness_relevancy_noRAG", run_rag=False)
