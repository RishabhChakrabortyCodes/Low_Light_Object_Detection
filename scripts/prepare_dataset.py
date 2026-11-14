import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import os

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
OUTPUT_PATH = f"data/outputs"

if not Path(OUTPUT_PATH).exists():
    os.mkdir(OUTPUT_PATH)

CLASSES_CSV_PATH = f"{OUTPUT_PATH}/classes.csv"
if Path(CLASSES_CSV_PATH).exists():
    print("The classes.csv file already exists...")
else:
    df.to_csv(CLASSES_CSV_PATH, index=False)

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
SPLIT_CSV_PATH = f"{OUTPUT_PATH}/splits"
if Path(SPLIT_CSV_PATH).exists():
    print("The split files already exists...")
else:
    os.mkdir(SPLIT_CSV_PATH)
    train_df.to_csv(f"{SPLIT_CSV_PATH}/train.csv", index=False)
    val_df.to_csv(f"{SPLIT_CSV_PATH}/validation.csv", index = False)
    test_df.to_csv(f"{SPLIT_CSV_PATH}/test.csv", index=False)



print(df.__len__)       

