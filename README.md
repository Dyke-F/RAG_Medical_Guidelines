# Medical Image In-Context Learning (ICL) with GPT-4V 
This repository is currently under construction. Usage might change in the future. 

## General Setup Instructions

Please follow the steps below:

1. **Python Installation**: Install Python from source. We used Python 3.11.6 throughout this project. 
2. **Dependency Installation**: Install necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
## Data Preparation

Place your dataset PDF files in accessible paths in the **Data** folder with a subfolder per entity. 

## Scripts

### Document comparison (GPT-4 + RAG)
Modify the configuration **.yaml** file. Set a variable documents_dir and the required topics.
Modify any other hyperparameters, like model or document split sizes as wished. 

#### Usage
Run the **compare_batch.py** script from the command line by specifying the path to your MIMIC ground truth data:
    
```bash
python3 compare_batch.py
```

You could also overwrite command line arguments here.

### Naive GPT-4
Configurations are loaded from the same **.yaml** file. If you wish to ask GPT-4 other questions, modify the *topics* list. 
    
```bash
python3 naive_gpt4.py
```

### Retrieval evaluation
Set the **.json** paths inside the **evaluations.py** file and run:
    
```bash
python3 evaluations.py
```
