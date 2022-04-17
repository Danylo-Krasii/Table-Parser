# Table-Parser

**Table-Parser** is a small project, the essence of which is to parse photos with tables on them and convert it to pandas dataframe and then to excel file.

This project uses more than just built-in modules, please check the `requirements.txt` file.
```
pip install -r requirements.txt
```

Usage example:

```python
from tab_parser import tabParser
from os import listdir
from os.path import join
import cv2

files_path = '.\Data'
save_to_path = '.\Data'

onlyfiles = [f for f in listdir(files_path) if ('.xlsx' not in f)] 
parser = tabParser()

for file in onlyfiles:
    img = cv2.imread(join(files_path, file), 0)
    df = parser.parse(img, join(save_to_path, file.replace('.jpg', '.xlsx')))
```
