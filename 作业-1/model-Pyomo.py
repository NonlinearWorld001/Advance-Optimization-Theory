from pyomo.environ import *
import numpy as np
import pandas as pd

data = pd.read_excel('作业-1\Assign1_data.xlsx', header=0, names=['y', 'x1', 'x2', 'x3'])

# print("数据列名:", data.columns.tolist())
# print("前3行数据:\n", data.head(3))

y = data.iloc[:, 0].values       # shape: (427,)
x0 = data.iloc[:, 1:4].values.T  # shape: (3,427)


# model = ConcreteModel()

# model.x = Var()
# model.y = Var(within=Integers)



