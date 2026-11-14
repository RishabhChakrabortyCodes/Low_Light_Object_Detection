import pandas as pd
from pathlib import Path
import torch 
from sklearn.model_selection import train_test_split

EXDARK_PATH = "data/raw/images/ExDark"
EXDARK_DIR = Path(EXDARK_PATH)

data = []

for directory in EXDARK_DIR.iterdir():
    class_name = str(directory).split('/')[4]
    count = 0
    for _file in directory.iterdir():
        count += 1
        file_data = {'image': _file, 'class': class_name}
        data.append(file_data)
    print(f"Number of images in class {class_name}: {count}")
        
df = pd.DataFrame(data)

df.to_csv(f"{EXDARK_PATH}/classes.csv", index=False)

# Train/Test/Val Split

# First split: 70% train, 30% temporary
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["class"]
)

# Second split: 15% validation, 15% test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["class"]
)

train_df.to_csv(f"{EXDARK_PATH}/train.csv", index=False)
val_df.to_csv(f"{EXDARK_PATH}/validation.csv", index = False)
test_df.to_csv(f"{EXDARK_PATH}/test.csv", index=False)

print(df.__len__)       

