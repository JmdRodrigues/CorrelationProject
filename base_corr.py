import pandas as pd
import matplotlib.pyplot as plt

db = pd.read_excel("Dahs_test/Base Dados_final.xlsx", "Base_1")
print(db)
plt.matshow(db.corr())
plt.show()


