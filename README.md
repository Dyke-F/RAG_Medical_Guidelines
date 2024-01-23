# Medical Image In-Context Learning (ICL) with GPT-4V 
Attention: This repository is currently under construction. Usage might change in the future. 

## General Setup Instructions

Please follow the steps below:

1. **Python Installation**: Install Python from source. We used Python 3.11.6 throughout this project. 
2. **Dependency Installation**: Install necessary dependencies. :
   ```bash
   pip install -r requirements.txt
   ```

3. **Repository Structure**:
.
├── Datafiles                           # contains subdirecotories for each dataset with .csv files containing paths to the test samples
│   ├── CRC100K
│   ├── MHIST
│   └── PCam
├── Figures
│   ├── CoverLetter Figure.pptx
│   ├── PPW
│   ├── Submission
│   └── embeddings
├── GPT4MedSubmission.code-workspace
├── Prompts
│   ├── CRC100K
│   ├── MHIST
│   └── PCam
├── README.md
├── Results
│   ├── CRC100K
│   ├── MHIST
│   └── PCam
├── Stats
│   ├── CRC100K
│   ├── MHIST
│   └── PCam
├── VisionModels
│   ├── CRC100K
│   ├── MHIST
│   ├── PCam
│   ├── create_embeddings.ipynb
│   ├── fewshot-histo
│   ├── make_pcam_imgs.ipynb
│   ├── run_finetune.ipynb
│   ├── run_nearest_neighbours.ipynb
│   ├── train_classifier.ipynb
│   └── venv
├── Visualisations
│   ├── CRC100K_eval
│   ├── MHIST_eval
│   ├── PCam_eval
│   ├── venv
│   └── visualisations
├── __pycache__
│   ├── dataset.cpython-311.pyc
│   ├── evaluate.cpython-311.pyc
│   ├── knn_dataset.cpython-311.pyc
│   ├── multi_image_knn_dataset.cpython-311.pyc
│   └── vision.cpython-311.pyc
├── config
│   ├── CRC100K
│   ├── MHIST
│   └── PCam
├── data
│   ├── CRC-VAL-HE-7K-png
│   ├── MHIST
│   └── PCam
├── dataset.py
├── evaluate.py
├── evaluate_for_publication.py
├── knn_dataset.py
├── main.py
├── make_datasets.ipynb
├── prepare_for_VisionModels.ipynb
├── requirements.txt
├── text_embeddings.ipynb
├── utils.py
├── venv
│   ├── bin
│   ├── include
│   ├── lib
│   ├── pyvenv.cfg
│   └── share
└── vision.py

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
