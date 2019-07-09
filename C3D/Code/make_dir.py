# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd

path = os.path.dirname(os.getcwd())
labels = pd.read_csv(os.path.join(path,'Label_Map','label.txt'))

for i in range(len(labels.Class_name)):
    
    path1 = os.path.join(path,'Data',labels.Class_name[i])
    path2 = os.path.join(path,'Raw_Data',labels.Class_name[i])
    
    if not (os.path.exists(path1) and os.path.exists(path2)):
        os.mkdir(path1)
        os.mkdir(path2)
    else:
        print('\nDirectory already exists, please delete it!\n')
        sys.exit(0)

