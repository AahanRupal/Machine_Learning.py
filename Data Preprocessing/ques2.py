import matplotlib.pyplot as plt
import pandas as pd

org_df=pd.read_csv("landslide_data_original.csv")
df= pd.read_csv("landslide_data_miss.csv")
org_df["dates"]=pd.to_datetime(df["dates"],dayfirst=True)
df["dates"]=pd.to_datetime(df["dates"],dayfirst=True)
df= df.dropna(subset='stationid')
df= df.drop(df[df.isna().sum(axis=1)>2].index)
nan=df.isna().sum()

df.reset_index(inplace=True,drop=True)
for i in (df.columns[2:]):
    for j in range(len(df)):
        if pd.isna(df.loc[j,i]):
            x=df.loc[j,df.columns[0]]
            k=j-1
            while(pd.isna(df.loc[k,i]) and k>=0):
                k-=1
            y1=df.loc[k,i]
            x1=df.loc[k,df.columns[0]]
            k=j+1
            while(pd.isna(df.loc[k,i]) and k<len(df)):
                k+=1
            y2=df.loc[k,i]
            x2=df.loc[k,df.columns[0]]
            df.loc[j,i]=(((y2-y1)*((x-x1)/(x2-x1))))+y1

df.to_csv("landslide_data_LI.csv")
print(df)
ip=pd.read_csv("landslide_data_LI.csv")

for col in (ip.columns[3:]):
    sort_c = sorted(ip[col])
    n=len(sort_c)-1
    sort_o=sorted(org_df[col])
    m=len(sort_o)-1
    if m%2==0:
        median1 = (sort_o[m//2+1]+sort_o[m//2])/2
    else:
        median1= sort_o[m+1//2]
    if n%2==0:
        median2 = (sort_c[n//2+1]+sort_c[n//2])/2
    else:
        median2 = sort_c[n+1//2]
    print(f"Median(org)={round(median1,2)}"+ f" Median(After IP)={round(median2,2)}")

    mean1=sum(org_df[col])/len(org_df[col])
    mean2= sum(ip[col])/len(ip[col])
    print(f"The statistical measures of {col} attribute are:")
    print(f"Mean(org)={round(mean1,2)}"+ f" Mean(After IP)={round(mean2,2)}")

    list_ip=[]
    list_org=[]
    for i in sort_o:
        difference = (i-mean1)**2
        list_org.append(difference)
    for j in sort_c:
        difference = (j-mean2)**2
        list_ip.append(difference)
    std1= (sum(list_org)/len(sort_o))**0.5
    std2 = (sum(list_ip)/len(sort_c))**0.5
    print(f"Std(org)={round(std1,2)}"+ f" Std(After IP)={round(std2,2)}")
    print("\n")

values=[]
for col in (df.columns[2:]):
    num=0
    for n in range(len(ip)):
        num+=(ip.at[n,col]-org_df[(org_df["dates"]==ip.at[n,"dates"]) & (org_df["stationid"]==ip.at[n,"stationid"])][col].values[0])**2
    values.append((num/nan[col])**0.5)
print("RMSE of each Attribute: ",values)

attri_L=['temperature','humidity','pressure','rain','lightavg','lightmax','moisture']
plt.plot(attri_L,values)
plt.title("RMSEs vs Attributes")
plt.xlabel("Attributes")
plt.ylabel("RMSE")
plt.show()

















