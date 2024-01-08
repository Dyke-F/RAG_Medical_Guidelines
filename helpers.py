import json

src = "/Users/dykeferber/Desktop/Guideline_LLM_Paper/Vector_LLM_for_Medical_QA-main/Results/GPT4_RAG/gpt-4_outputs.json"

def read_json_file(file_path):
    with open(file_path, "r") as f:
        content = f.read()
    json_data_list = json.loads(f'[{content.replace("}{", "},{")}]')
    with open(file_path, "w") as f:
        json.dump(json_data_list, f)

read_json_file(src)