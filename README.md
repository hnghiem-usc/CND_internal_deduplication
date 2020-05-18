# CND_internal_deduplication
Code to identify internal duplicates from ChoiceMaker extracts.

## REQUIREMENTS 
Make sure you already have the following packages/libraries using either ANACONDA or pip:
* numpy
* panda
* networkx
* itertoos 
* csv

Also place the corresponding extract csv file in the same directory as the code files.
This code was initially written for the JCATS-CWS linkage.
extract_08751_2020_edited.csv

## RUN
On a MAC, you can run the python (.py) directly from Terminal.
Note: change the filename to match yours if necessary.
```
cd CODE_DIRECTORY_WHERE_FILES_ARE
python3 assign_cycle.py -ext extract_08751_2020_edited.csv -key JCATS
```

Or you can run from Jupyter notebook/lab with the following commands:
```python
import numpy as np 
import pandas as pd 
import networkx as nx
import csv
import argparse 
from itertools import combinations
%run -i "assign_cycle.py" -ext extract_08751_2020_edited.csv -key JCATS
```
