import pandas as pd
import numpy as num
import sys

species = set()
pd_iris = pd.read_csv('Iris.csv')
rows, columns = pd_iris.shape
for i in range(rows):
    species.add(pd_iris.loc[i, 'Species'])
species = list(species)
for i in range(rows):
    for s in species:
        if pd_iris.loc[i, 'Species'] == s:
            pd_iris.loc[i, 'Species'] = species.index(s)

pd_iris.to_csv('IRIS2.csv')