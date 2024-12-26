# IBM Transactions for Anti-Money Laundering (AML) Dataset  

This project uses the IBM Transactions for Anti-Money Laundering (AML) dataset. The dataset can be downloaded from Kaggle and integrated into the project by following the steps below.  

## 1. Download the Dataset  

### Method 1: Python Script for Automatic Download  

```python
import kagglehub

# Download the dataset
path = kagglehub.dataset_download("ealtman2019/ibm-transactions-for-anti-money-laundering-aml")

print("Path to dataset files:", path)
```

Move the downloaded file to this directory:  

```bash
mv ~/Downloads/HI-Small_Trans.csv data/
```

### Method 2: Command Line Download  

Ensure your Kaggle API key is set up (place the API key JSON file in: `~/.kaggle/kaggle.json`).  

```bash
# Download the dataset
kaggle datasets download -d ealtman2019/ibm-transactions-for-anti-money-laundering-aml -p data/

# Unzip the downloaded file
unzip data/ibm-transactions-for-anti-money-laundering-aml.zip -d data/
```

Ensure the file is located here:  

```bash
data/HI-Small_Trans.csv
```

---

## 2. Verify the File Location  

The file should be placed in the following directory:  

```bash
project-root/
    data/
        HI-Small_Trans.csv
```

To verify the file, run the following script:  

```python
import os
print("File exists?", os.path.exists('data/HI-Small_Trans.csv'))
```

---

## 3. Important Notes  

```markdown
- The dataset is large (approximately 450 MB), so it has not been included in this repository.  
- After downloading, place the files in this folder. The project will automatically detect the dataset.  
- For more details, please check the main README.md file in the root directory.  
```

---

## 4. License and References  

```markdown
- [Kaggle Dataset Link](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)  
- License: Community Data License Agreement  
```
