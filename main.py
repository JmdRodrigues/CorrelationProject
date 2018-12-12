import numpy as np
import pandas as pd
import openpyxl as xl
from itertools import islice

s = xl.load_workbook("Base_Dados_final.xlsx", data_only=True)
ws = s['Base de dados']

# create Dataframe

data = ws.values
cols = next(data)[1:]
data = list(data)
idx = [r[0] for r in data]
data = (islice(r, 1, None) for r in data)
df = pd.DataFrame(data, index=idx, columns=cols)

# print(df)

print(df['Idade'][355])


