import json
import textwrap
from pathlib import Path

def find_json_files(p):
    # returns an iterator 
    return Path(p).glob("*.json")

def read_json_file(file_path):
    with open(file_path, "r") as f:
        content = f.read()
    # stupid workaround because multiple json objects are stored as {}{}{} in the file
    json_data_list = json.loads(f'[{content.replace("}{", "},{")}]')

    return json_data_list

def store_formatted_txt(json_data, file_path, max_line_length=80):
    with open(file_path, 'a') as f:
        for i, data in enumerate(json_data, 0):
            f.write(f"--- New Conversation: #{i} ---".center(100, "-"))
            f.write("\n")
            f.write("# timestamp: " + str(data.get("# timestamp", "No timestamp available")) + "\n")
            f.write("# document chunk size: " + str(data.get("# chunk_size", "No chunk size available")) + "\n")
            f.write("# document split overlap: " + str(data.get("# overlap", "No overlap available")) + "\n")
            f.write("\n")
            #f.write(textwrap.fill("Clinician: What do the provided documents say about "
            #                      + data.get("Human Message", [])[0].strip(), max_line_length) + "?\n")
            f.write("Clinician: What do the provided documents say about "
                                  + data.get("Human Message", [])[0].strip() + "?\n")

            f.write("GPT-4: ")
            raw_text = data.get("AI Response", [])[0]
            paragraphs = raw_text.split("\n")
            
            #for paragraph in paragraphs:
            #    wrapped_text = textwrap.fill(paragraph, max_line_length)
            #    f.write(wrapped_text + "\n")
            for paragraph in paragraphs:
                f.write(paragraph + "\n")

            f.write("\n\n")

def main():
    files = find_json_files("/Users/dykeferber/Desktop/Guideline_LLM_Paper/Vector_LLM_for_Medical_QA-main/NaiveGPT4Results")
    for file_path in files:
        json_data = read_json_file(file_path)
        store_formatted_txt(json_data, Path(file_path.stem + "_unstructured.txt"))

if __name__ == "__main__":
    main()
