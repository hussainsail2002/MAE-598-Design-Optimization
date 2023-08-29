from scipy.optimize import minimize

def objective(x):
    x1=x[0]
    x2=x[1]
    x3=x[2]
    x4=x[3]
    x5=x[4]
    obj_1= (x1-x2)**2+(x2+x3-2)**2+(x4-1)**2+(x5-1)**2
    return obj_1

def constraint1(x):
    return x[0]+(3*x[1])

def constraint2(x):
    return x[2]+x[3]-(2*x[4])

def constraint3(x):
    return x[1]-x[4]

x0=[1,-1,-6,5,9]

b=(-10,10)
bnds=(b,b,b,b,b)
con1={'type':'eq','fun':constraint1}
con2={'type':'eq','fun':constraint2}
con3={'type':'eq','fun':constraint3}
cons=[con1,con2,con3]

sol = minimize(objective,x0,method='SLSQP',bounds=bnds,constraints=cons)
print(sol)