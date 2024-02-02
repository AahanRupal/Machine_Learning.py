import matplotlib.pyplot as plt
import pandas as pd

ip = pd.read_csv("landslide_data_LI.csv")    

dscrpxn=ip.describe()
for j in (ip.columns[3:]):
    lower=dscrpxn[j]['25%']
    upper=dscrpxn[j]['75%']
    out=[]
    interquartile_range = upper-lower
    for i in ip[j]:
        if(not (lower - 1.5*interquartile_range < i < upper +1.5*interquartile_range)):
            out.append(i)
    print(f'{j}:',out)
    print(len(out))
    plt.boxplot(ip[j])
    plt.title(f"boxplot({j})")
    plt.show()

for i in (ip.columns[3:]):
    lower=dscrpxn[i]['25%']
    upper=dscrpxn[i]['75%']
    out=[]
    interquartile_range = upper-lower
    for j in range(len(ip[i])):
        sorted_file = sorted(ip[i])
        m=len(sorted_file)-1
        if m%2==0:
            median=(sorted_file[m//2+1]+sorted_file[m//2])/2
        else:
            median=sorted_file[m+1//2]
        if(not (lower - 1.5*interquartile_range < ip.at[j,i] < upper +1.5*interquartile_range)):
            ip.at[j,i]=median
            out.append(ip.at[j,i])
    print(f'{i}:',out)
    plt.boxplot(ip[i])
    plt.title(f"boxplot(replaced)({i})")
    plt.show()

ip.to_csv("Corrected.csv",index=False)












