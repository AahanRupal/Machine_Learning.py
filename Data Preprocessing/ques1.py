import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df= pd.read_csv('landslide_data_original.csv')

min = np.min(df['temperature'])
max = np.max(df['temperature'])
print("The statistical measures of Temperature attribute are:")
print(f"Min={round(min,2)}")
print(f"Max={round(max,2)}")

mean= sum(df['temperature'])/len(df['temperature'])
print(f"mean={round(mean,2)}")

sorted_temp = sorted(df['temperature'])

l=[]
for j in sorted_temp:
    diff = (j-mean)**2
    l.append(diff)
std = (sum(l)/len(sorted_temp))**0.5
print(f"Std={round(std,2)}")

m=len(sorted_temp)-1
if m%2==0:
    median = (sorted_temp[m//2+1]+sorted_temp[m//2])/2
else:
    median = sorted_temp[m+1//2]
print(f"median={round(median,2)}")


matrix=np.zeros((7,7))
def correlation(x,y):
    xm=np.mean(x)
    ym=np.mean(y)
    var_1=0
    var_2=0
    var_3=0
    for i in range(len(x)):
        var_1+=(x[i]-xm)*(y[i]-ym)
        var_2+=(x[i]-xm)**2
        var_3+=(y[i]-ym)**2
    r= var_1/((var_2*var_3)**0.5)
    return r

temperature=list(df['temperature'])
humidity=list(df['humidity'])
pressure=list(df['pressure'])
rain=list(df['rain'])
lightavg=list(df['lightavg'])
lightmax=list(df['lightmax'])
moisture=list(df['moisture'])

attri=[temperature,humidity,pressure,rain,lightavg,lightmax,moisture]
attri_L=['temperature','humidity','pressure','rain','lightavg','lightmax','moisture']

for m in range(7):
    for n in range (7):
        tr=correlation(attri[m],attri[n])
        matrix[m][n]=round(tr,2)
print('\n')

correlated_matrix=pd.DataFrame(matrix,index=attri_L,columns=attri_L)
print(correlated_matrix)
print("\n")
print("light max is redundant attribute")
print("\n")

x=df.groupby('stationid').get_group('t12')
plt.hist(x['humidity'],bins=5)
plt.title('Humidity (stationid=t12)')
plt.xlabel('Humidity')
plt.ylabel('Count')
plt.show()


    










