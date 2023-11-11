import numpy as np
import matplotlib.pyplot as plt
from cvxopt import solvers, matrix

def objective(x):
    f= x[0,0]**2 + (x[1,0]-3)**2
    return f

def objectiveg(x):
    df=np.array([2*x[0,0],2*x[1,0]-6])
    return df

def constraint(x):
    g=np.array([[x[1,0]**2-2*x[0,0]],[(x[1,0]-1)**2+5*x[0,0]-15]])
    return g
    
def constraintg(x):
    dg = np.array([[-2, 2*x[1,0]],[5,2*x[1,0]-2]])
    return dg

def linesearch(s,w_old,mu_old,x):
    mu_old=np.array(mu_old).reshape(-1,1)
    w_old=np.array(w_old).reshape(-1,1)
    t=0.1
    b=0.8
    a=1
    D=np.array(s)
    w=np.maximum(abs(mu_old),0.5*(w_old+abs(mu_old)))
    
    count = 0
    while count<100:
        g_x=-1*constraint(x+a*D)
        g_x1=-1*constraint(x)
        g_x2=constraint(x)
        phi_a=objective(x+a*D) + np.matmul(w.T,abs(np.where(g_x<0,g_x,0)))
        phi0 = objective(x) + np.matmul(w.T,abs(np.where(g_x1<0,g_x1,0)))
        dphi0 = np.matmul(objectiveg(x),D) + np.matmul(w.T,(np.matmul(constraintg(x),D)*np.where(g_x2>0,1,0)))
        psi_a=phi0 + t*a*dphi0
        if phi_a < psi_a:
            break
        else:
            a=a*b
            count=count+1
    return a,w
    
def mysqp(x):
    x_cur=x
    solution = x_cur
    W = matrix(np.identity(len(constraint(x)))) # hessian initialized as a identity
    mu_old=np.zeros(len(constraint(x))) # vector of lagrangian for inequality constraints
    w=np.zeros(len(constraint(x)))
    gnorm=np.linalg.norm(objectiveg(x)+ np.matmul(mu_old.T,constraintg(x)))
    A=matrix(np.matrix([0.0,0.0]))
    b=matrix(np.array([0.0]))
    count1=0
    while gnorm>0.07:
        df=matrix((objectiveg(x_cur)).T)*1.0
        dg=matrix(constraintg(x_cur))*1.0
        g=matrix(constraint(x_cur)*-1)*1.0
        x1=solvers.qp(W,df,dg,g,A,b,kktsolver='ldl', options={'kktreg':1e-9})
        mu_new=x1['z']
        s=x1['x']
        a,w=linesearch(s,w,mu_old,x_cur)
        dx=a*s
        x_cur = x_cur + dx
        y_k=matrix(objectiveg(x_cur),(1,2)) + mu_new.T*matrix(constraintg(x_cur)) - matrix(objectiveg(x_cur-dx),(1,2)) - mu_new.T*matrix(constraintg(x_cur-dx))
        y_k=y_k.T
        
        if np.array(dx.T*y_k) >= np.array(0.2*dx.T*W*dx):
            theta = 1
        else:
            theta = (0.8*dx.T*W*dx)/(dx.T*W*dx-dx.T*y_k)

        dg_k = theta*y_k+(1-theta)*W*dx
        W=W+(dg_k*dg_k.T)/(dg_k.T*dx)
        gnorm = np.linalg.norm(matrix(objectiveg(x_cur),(1,2))+mu_new.T*matrix(constraintg(x_cur)))
        count1=count1+1
        mu_old=mu_new

        solution=np.c_[solution,x_cur]
        
    return solution

def report(sol_fin):
    x2=sol_fin
    r,iter_1=np.shape(x2)
    x_axis=np.arange(0,iter_1)
    y_axis=np.zeros(len(x_axis))
    for i in range(iter_1):
        y_axis[i]= objective(np.array(x2[:,i]).reshape(-1,1))
    
    fig,(ax1,ax2)=plt.subplots(1,2, figsize=(8,8))
    y_axis1=np.log(y_axis-y_axis[-1]+np.spacing(1))
    ax1.plot(x_axis,y_axis1,color='k')

    x_cont_1=np.arange(start=-10,stop=10,step=0.5)
    y_cont_1=np.arange(start=-10,stop=10,step=0.5)

    f_x=np.zeros([len(x_cont_1),len(y_cont_1)],dtype=int)
    for i in range(len(x_cont_1)):
        for j in range(len(y_cont_1)):
            f_x[i][j]=x_cont_1[j]**2 + (y_cont_1[i]-3)**2

    x_cont_2=np.arange(start=-4,stop=4,step=0.5)
    y_cont_2=np.arange(start=-4,stop=4,step=0.5)

    g_x=np.zeros([len(x_cont_2)])
    for i in range(len(x_cont_2)):
        g_x[i]=x_cont_2[i]**2/2

    x_cont_3=np.arange(start=-7,stop=9,step=0.5)
    g_x1=np.zeros([len(x_cont_3)])
    for i in range(len(x_cont_3)):
        g_x1[i]= (15 - (x_cont_3[i]-1)**2)/5
        


    ax2.contourf(x_cont_1,y_cont_1,f_x,30)
    ax2.plot(g_x,x_cont_2)
    ax2.plot(g_x1,x_cont_3)
    ax2.plot(0,3, marker="o", markersize=5,markeredgecolor="red")
    ax2.plot(x2[0],x2[1],marker='o',markersize='5',linestyle='-',color='b')
    ax2.plot(x2[0,-1],x2[1,-1],marker='o',markersize='6',markeredgecolor="red")
    plt.show()


x0=np.array([[1],[1]])
sol=mysqp(x0)
report(sol)


