import pandas as pd
import  numpy as np
from matplotlib import pyplot as plt
import sklearn.model_selection
from collections import Counter 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

data=pd.read_csv('Iris.csv')
mat=data.drop(columns=['Species'] )
# print(mat)
y=data['Species']
y=np.array(y)

described=mat.describe()
for j in (mat.columns[:]):
    median=mat[j].median()
    lower=described[j]['25%']
    upper=described[j]['75%']
    iqr=upper-lower
    for i in range(len(mat[j])):
        if(not(lower-1.5*iqr<mat.at[i,j]<upper+1.5*iqr)):
            mat.at[i,j]=median
            
Xog=mat.copy()
# print(Xog)
            
for j in (Xog.columns[:]):
    mean=Xog[j].mean()
    for i in range(len(Xog[j])):
        Xog.at[i,j]=Xog.at[i,j]-mean
        
X=Xog.copy()
Xt=X.T
C=np.dot(Xt,X)
# print(C)

eigenval,eigenvect=np.linalg.eig(C)
eigenval=np.sort(eigenval)[::-1]

selected_vect=eigenvect[:,:2]
Q=selected_vect

projection=np.dot(mat,Q)
eigenVector1=eigenvect[:,0]
eigenVector2=eigenvect[:,1]

plt.scatter(projection[0:,0],projection[0:,1], c='blue',label='Projection')
plt.quiver(5,-5.5,eigenVector1[0]*1000,eigenVector1[1]*1000,color='red',label='EigenVector 1')
plt.quiver(5,-5.5,eigenVector2[0]*1000,eigenVector2[1]*1000,color='green',label='EigenVector 2')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.grid(True)
# plt.show()

Reconst_mat=np.dot(projection,Q.T)

# print(Reconst_mat)
values=[]
p=0
for col in (mat.columns[:]):
    num=0
    for n in range(len(mat)):
        num+=(mat.at[n,col]-Reconst_mat[n,p])**2
    values.append((num/len(mat))**0.5)
    p+=1
# print("RMSE of each Attribute: ",values)

#part2

projection_train,projection_test,y_train,y_test=sklearn.model_selection.train_test_split(projection,y,random_state=104,test_size=0.20,shuffle=True)
# print(projection_test)
# print(projection_train)
K=5
actual_predicted=[]
for k in range(len(projection_test)):
    distance=[]
    for z in range(len(projection_train)):
        distance_=np.sqrt(np.sum((projection_test[k]-projection_train[z])**2))
        distance.append([distance_,y_train[z]])

    distance.sort()
    nearest_distance=distance[:K]
    
    predicted_class = Counter(top_K_neighbor[1] for top_K_neighbor in nearest_distance).most_common(1)[0][0]
    actual_predicted.append((y_test[k],predicted_class))
    
matrix=np.array(actual_predicted)
# print(matrix)

actual=matrix[:,0]
predicted=matrix[:,1]
c_matrix=confusion_matrix(actual,predicted)
print("Confusion Matrix: ")
print(c_matrix)
ConfusionMatrixDisplay(c_matrix,display_labels=['Class 1','Class 2','Class 3']).plot()
plt.show()