import pandas as pd
from pathlib import Path

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

print(df.__len__)       

