import numpy as np
import pandas as pd 
outlier_corrected=pd.read_csv("Corrected.csv")

new_df=outlier_corrected.copy()
for c in (outlier_corrected.columns[3:]):
    max_xi=max(outlier_corrected[c])
    min_xi=min(outlier_corrected[c])
    for d in range(len(outlier_corrected[c])):
        normalised_x= ((((outlier_corrected.at[d,c]-min_xi)/(max_xi-min_xi))*(12-5))+5)
        new_df.at[d,c]=normalised_x
print(new_df.head())

print("\nNormalized:")
for col in (new_df.columns[3:]):
    min_=min(new_df[col])
    max_=max(new_df[col])
    print(f'{col} Minimum={min_}, Maximum={max_}')

print("\noriginal:")
for col in (outlier_corrected.columns[3:]):
    min_=min(outlier_corrected[col])
    max_=max(outlier_corrected[col])
    print(f'{col} Minimum={round(min_,2)}, Maximum={round(max_,2)}')

new_df=outlier_corrected.copy()
for c in (outlier_corrected.columns[3:]):
    mean=np.mean(outlier_corrected[c])
    std=np.std(outlier_corrected[c])
    for d in range(len(outlier_corrected[c])):
        standardised_x= (outlier_corrected.at[d,c]-mean)/std
        new_df.at[d,c]=standardised_x
print(new_df.head())


print("\nStandardized:")
for col in (outlier_corrected.columns[3:]):
    std=np.std(new_df[col])
    mean=np.mean(new_df[col])
    print(f'{col} Mean={round(mean,2)}, Std={round(std,2)}')

print("\noriginal:")
for col in (outlier_corrected.columns[3:]):
    std=np.std(outlier_corrected[col])
    mean=np.mean(outlier_corrected[col])
    print(f'{col} Mean={round(mean,2)}, Std={round(std,2)}')






            
