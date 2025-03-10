from pyomo.environ import *
import pandas as pd
from copt_pyomo import *

data = pd.read_excel('assignment-1\Assign1_data.xlsx', header=0, names=['y', 'x1', 'x2', 'x3'])

# print("数据列名:", data.columns.tolist())
# print("前3行数据:\n", data.head(3))

y = data.iloc[:, 0].values       # shape: (427,)
x0 = data.iloc[:, 1:4].values.T  # shape: (3,427)

w1_fixed = [0.5767, 0.1875, -0.1472, 0.1251]
b11_fixed = -0.0895
M = 10000000 # big M


model = ConcreteModel()

# hiden layer: weight and bias
model.w0 = Var(range(4), range(3), within=Reals)
model.b0 = Var(range(4), within=Reals)

# Integer variable
model.delta = Var(range(4), range(427), within=Binary)
# constraint variable
model.h = Var(range(4), range(427), within=NonNegativeReals)
model.z = Var(range(4), range(427), within=Reals)

# objective function
def obj_func(model):
    total_error = 0
    for k in range(427):
        y_pred = sum(w1_fixed[j] * model.h[j, k] for j in range(4)) + b11_fixed
        total_error += (y[k] - y_pred) ** 2
    return total_error

# constraints
# z_cons: z = sum(w0 * x0 + b0)
def z_constraint_rule(model, j, k):
    return model.z[j,k] == sum(model.w0[j,i] * x0[i][k] for i in range(3)) + model.b0[j]
model.z_cons = Constraint(range(4), range(427), rule=z_constraint_rule)

# ReLU_cons1: h > z
def ReLU_constraint_rule1(model, j, k):
    return model.h[j,k] >= model.z[j,k]
model.ReLU_cons1 = Constraint(range(4), range(427), rule=ReLU_constraint_rule1)

# ReLU_cons2: h >= 0
def ReLU_constraint_rule2(model, j, k): 
    return model.h[j,k] >= 0
model.ReLU_cons2 = Constraint(range(4), range(427), rule=ReLU_constraint_rule2)

# ReLU_cons3: h < z + (1-delta) * M
def ReLU_constraint_rule3(model, j, k):
    return model.h[j,k] <= model.z[j,k] + model.delta[j,k] * M
model.ReLU_cons3 = Constraint(range(4), range(427), rule=ReLU_constraint_rule3)

# ReLU_cons4: h <= M * delta
def ReLU_constraint_rule4(model, j, k):
    return model.h[j,k] <= M * (1 - model.delta[j,k])
model.ReLU_cons4 = Constraint(range(4), range(427), rule=ReLU_constraint_rule4)

# set objective function
model.obj = Objective(rule=obj_func, sense=minimize)

# solve model
solver = SolverFactory('copt_direct')
solver.options['TimeLimit'] = 100
result = solver.solve(model, tee=True)

# print results
if result.solver.status == SolverStatus.ok:
    print("求解成功，全局最优目标函数值为：\n", model.obj())
    print("全局最优解为：\n")
    for i in range(4):
        for j in range(3):
            print(f"w0[{i},{j}] = {value(model.w0[i,j])}")
    for i in range(4):
        print(f"b0[{i}] = {value(model.b0[i])}")    
else:
    print("未找到全局最优解")
    print("当前解为：\n")
    for i in range(4):
        for j in range(3):
            print(f"w0[{i},{j}] = {value(model.w0[i,j])}")
    for i in range(4):
        print(f"b0[{i}] = {value(model.b0[i])}")
    print("当前解的目标函数值为：\n", model.obj())
    print("下界为：\n", result.problem.lower_bound)
