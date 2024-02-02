import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('Iris.csv')

#Que.I.a)
trueClassLables=list(df['Species'])
attribute=df.columns[:4]
df_independent=df[attribute]
matrix=df_independent.values

#Que.I.b)
def outliers(columnname,df):
    df_new = df.copy()
    df_new[columnname] = pd.to_numeric(df_new[columnname], errors='coerce')

    Q1 = np.percentile(df_new[columnname], 25)
    Q2=np.percentile(df_new[columnname],50)
    Q3 = np.percentile(df_new[columnname], 75)
    IQR=Q3-Q1

    lower_bound=(Q1-(1.5*IQR))
    upper_bound=(Q3+(1.5*IQR))
    
    l=[]
    for i in range(len(df_new[columnname])):
        if df_new.iloc[i][columnname]< lower_bound or df_new.iloc[i][columnname]>upper_bound:
            l.append(df_new.iloc[i][columnname])
            df_new.loc[i, columnname] = Q2

    return df_new

X=outliers('SepalLengthCm',df_independent)
X=outliers('SepalWidthCm',X)
X=outliers('PetalLengthCm',X)
X=outliers('PetalWidthCm',X)

#Que.I.c)
def subMean(columnname,file):
    df_new=file.copy()
    mean=np.mean(file[columnname])
    for i in range(len(file[columnname])):
        m=file.iloc[i][columnname]-mean
        df_new.loc[i, columnname] = m
    return df_new

X1=subMean('SepalLengthCm',df_independent)
X1=subMean('SepalWidthCm',X1)
X1=subMean('PetalLengthCm',X1)
X1=subMean('PetalWidthCm',X1)

matrix_X1=X1.values

def correlation_matrix(x):
    C=np.corrcoef(x.T)
    return C

C=correlation_matrix(matrix_X1)

def eigenAnalysis(data):
    eigenvalues,eigenvectors=np.linalg.eig(data)
    return eigenvalues, eigenvectors
evalues, evectors=eigenAnalysis(C)

def orderEigenValues(data):
    sorted=np.sort(evalues)[::-1]
    return sorted
sortEigenVal= orderEigenValues(evalues)

evectors=evectors[:,np.argsort(sortEigenVal)[:2]]
Q=evectors

reduced_data = np.dot(matrix_X1, Q)
# print(reduced_data)


#Que.I.d)
plt.scatter(reduced_data[:,0],reduced_data[:,1],color='Orange')
eigenVector1=evectors[:,0]
eigenVector2=evectors[:,1]
scale=0.5
plt.quiver(0,0,eigenVector1[0]*scale,eigenVector1[1]*scale,color='red',label='EigenVector 1')
plt.quiver(0,0,eigenVector2[0]*scale,eigenVector2[1]*scale,color='green',label='EigenVector 2')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title("Scatter Plot of the First Two Principal Components with Eigen Directions")
plt.legend()
plt.show()

#Que.I.e)
X_reconstructed=np.dot(reduced_data,Q.T)
reconstucted_x=pd.DataFrame(X_reconstructed,columns=attribute)
# print(X)
print(reconstucted_x)

#Que.I.f)
def rmse(columnname):
    l=[]
    for i in range(len(X.columns)):
        m = (X.iloc[i][columnname] - reconstucted_x.iloc[i][columnname]) ** 2
        l.append(m)
    val=(sum(l)/len(l))**(1/2)
    return val
rmsee=[]
for i in X.columns:
    m=rmse(i)
    rmsee.append(m)
    
# print("RMSE values of respective attributes: ",rmsee)