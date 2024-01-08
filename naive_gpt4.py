import openai
import json
import os
from getpass import getpass
from dotenv import load_dotenv
from pathlib import Path
import hydra
from openai import OpenAI
from hydra.core.hydra_config import HydraConfig
from fastapi.encoders import jsonable_encoder


SYSTEM_MSG = """
You are an AI medical assistant specializing in oncology. Based on what you have learned from medical oncology guidelines, provide detailed and truthful information in response to inquiries from a medical doctor. Ensure your responses are:
- Relevant to the given context.
    For instance, when asked about chemoradiation, do not include information about chemotherapy alone.
- Presented in concise bullet points.
- Honest, especially when the answer isn't available in the guidelines that you have learned about. 
- Include citations and references.
- As detailed as possible. Include all details regarding patient and tumor characteristics like R-status (R0/R1), tumor grade and Tumor Stage (TNM).
- Include references to clinical trials (by name and not by number), survival data, exact treatments, their outcomes and details on the trial design. 
Based on the American (ASCO) and European (EMSO) medical oncology guidelines, what does the association say about the topic presented in the context?
"""

USER_MSG = """
You are a dedicated AI assistant providing detailed responses to a medical doctor. Your answers are based on the provided documents and are strictly truthful. If the information is not available to you, state it clearly. Refrain from including irrelevant or out-of-context details.
During your training you might have learned about specific cancer-related information extracted from both the ESMO and ASCO guidelines. Your task is to conduct a line-by-line comparison to identify and extract the similarities and differences between the two sets of guidelines.
The main objective is to pinpoint discrepancies in the recommendations.

It is important to consider all available details, for example including resection status (R0 vs R1), tumor stage etc. to allow for a correct comparison. 
Also, provide all the details from clinical trials, like the trial name, survival data, and the overall conclusion from the trial

Your structured output should be in the format:
    Comparison of (topic of the question) between ASCO and ESMO:
    Similarities:
        - (topic): ...
        - (topic): ...
    Differences:
        - (topic): ...
        - (topic): ...

Every subpoint in similarities and differences should be structured based on a useful (topic) as given in the data.
    For example: If recommendations can be seperated into adjuvant / locally advanced / metastatic disease, use these as topic and compare what the different institutions recommend. 
    For example: If different treatment options are given like surgery, radiation, chemotherapy, seperate your structured output by these.
    
Ensure all relevant details are given in your answer: This includes for instance:
    Names of clinical trials, the trial design, their outcomes and conclusions. 
    Specific patient and treatment characteristics that are compared (tumor stage, R0/R1, treatment details (timing, duration, substances)) 

Finally, summarize your comparison.
The given topic is: What do the guidelines on {guideline_topic} say about {topic}?
"""

topic_map = {
    "hcc_config": "Hepatocellular Carcinoma (HCC)",
    "pancreatic_config": "Pancreatic Cancer",
    "mCRC_config": "Metastatic Colorectal Cancer (mCRC)",
}

@hydra.main(config_path="./conf", config_name="mCRC_config", version_base="1.3")
def main(cfg):
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY") or getpass(
        f"Enter a valid OpenAI API key: "
    )
    assert os.environ["OPENAI_API_KEY"].startswith("sk-"), f"Invalid OpenAI API key"
    client = OpenAI()

    config_name = HydraConfig.get().job.config_name
    guideline_topic = topic_map[config_name]

    all_responses = []
    for topic in cfg.topics:
        COMPLETE_USER_MSG = USER_MSG.format(topic=topic, guideline_topic=guideline_topic)
        messages = [
            {
                "role": "system",
                "content": SYSTEM_MSG,
            },
            {
                "role": "user",
                "content": COMPLETE_USER_MSG,
            }
        ]

        response = client.chat.completions.create(
            messages=messages,
            model="gpt-4-0613",
            temperature=0,
        )

        all_responses.append(
            {"guideline_topic": guideline_topic,
             "topic": topic,
             "sys_msg": SYSTEM_MSG,
             "user_msg": COMPLETE_USER_MSG,
             "response": response.choices[0].message.content}
        )

    with open(f"{config_name}_naive_gpt4.json", "w") as f:
        json.dump(all_responses, f, indent=4)

if __name__ == "__main__":
    main()
