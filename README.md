## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

```python

import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
<img width="562" height="591" alt="Screenshot 2025-09-29 104402" src="https://github.com/user-attachments/assets/ec1f346f-3b0a-46c0-bba8-4177bd053711" />


```py
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
<img width="809" height="575" alt="Screenshot 2025-09-29 104408" src="https://github.com/user-attachments/assets/c64d0bb4-d500-45dc-bf7b-06cb333b52ba" />


```py
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```

<img width="545" height="538" alt="Screenshot 2025-09-29 104416" src="https://github.com/user-attachments/assets/43bff635-91ff-4765-bda3-5d26a8f41eb3" />

```py
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```

<img width="529" height="579" alt="Screenshot 2025-09-29 104422" src="https://github.com/user-attachments/assets/659b3614-d29a-4583-bd4b-767bb4ac7505" />

```py
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```
<img width="626" height="621" alt="Screenshot 2025-09-29 104428" src="https://github.com/user-attachments/assets/eb6cf820-5dae-4d67-ae74-43ac5d730436" />


```py
df2=pd.concat([df2,enc],axis=1)
df2
```




```py
pd.get_dummies(df2,columns=["nom_0"])
```
<img width="919" height="519" alt="Screenshot 2025-09-29 104434" src="https://github.com/user-attachments/assets/d806c53c-bf1d-4448-9e57-b82aee9f261d" />


```py
pip install --upgrade category_encoders
```
<img width="824" height="501" alt="Screenshot 2025-09-29 104453" src="https://github.com/user-attachments/assets/0bcc74f9-767e-4801-9050-e7875f4675e8" />


```py
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```
<img width="722" height="560" alt="Screenshot 2025-09-29 104458" src="https://github.com/user-attachments/assets/9f5fb556-e654-4d32-81aa-8776e5161c32" />


```py
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```
<img width="739" height="551" alt="Screenshot 2025-09-29 104503" src="https://github.com/user-attachments/assets/342ebedb-c68c-4d37-96b0-aa5225de2b55" />


```py
dfb=pd.concat([df,nd],axis=1)
dfb
```
<img width="953" height="530" alt="Screenshot 2025-09-29 104510" src="https://github.com/user-attachments/assets/ed8df796-570b-4b06-bde1-dcf333643bcc" />



```py
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
<img width="795" height="614" alt="Screenshot 2025-09-29 104517" src="https://github.com/user-attachments/assets/c78fb603-d69e-4768-a6ba-63d1611176b7" />


```py
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
<img width="1065" height="676" alt="Screenshot 2025-09-29 104524" src="https://github.com/user-attachments/assets/62dda11b-1be6-4f88-9058-a1c85a6aa14d" />


```py
df.skew()
```
<img width="473" height="332" alt="Screenshot 2025-09-29 104530" src="https://github.com/user-attachments/assets/026d0f17-c383-4c11-a31c-f2a568aa9009" />


```py
np.log(df["Highly Positive Skew"])
```
<img width="564" height="633" alt="Screenshot 2025-09-29 104536" src="https://github.com/user-attachments/assets/1b591c23-17ee-4343-abb2-5f79edd8b900" />


```py
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="557" height="604" alt="Screenshot 2025-09-29 104543" src="https://github.com/user-attachments/assets/ed626ee7-ebf2-4f5c-8316-69d50e12e41d" />



```py
np.sqrt(df["Highly Positive Skew"])
```
<img width="518" height="657" alt="Screenshot 2025-09-29 104549" src="https://github.com/user-attachments/assets/3fe47074-d567-4573-b241-165586c235b8" />


```py
np.square(df["Highly Positive Skew"])
```

<img width="530" height="637" alt="Screenshot 2025-09-29 104554" src="https://github.com/user-attachments/assets/70e61f64-b6cf-4d29-a7e0-e99234a0fdb0" />



```py
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
<img width="1353" height="611" alt="Screenshot 2025-09-29 104600" src="https://github.com/user-attachments/assets/bbc25c2c-09ef-4ed1-b68a-f1fc953c8a69" />


```py
df.skew()
```
<img width="647" height="399" alt="Screenshot 2025-09-29 104604" src="https://github.com/user-attachments/assets/526543ae-6e60-4a76-b901-1a86b2768295" />


```py
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
<img width="990" height="413" alt="Screenshot 2025-09-29 104610" src="https://github.com/user-attachments/assets/3d01ce60-5977-4514-a624-5186918d374f" />


```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
<img width="1369" height="679" alt="Screenshot 2025-09-29 104616" src="https://github.com/user-attachments/assets/6941e084-610d-40bf-9ed3-652d426823df" />

```py
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="863" height="721" alt="Screenshot 2025-09-29 104625" src="https://github.com/user-attachments/assets/b1b8f1d4-a754-4ba3-886d-f31c945234a3" />


```py
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

<img width="924" height="639" alt="Screenshot 2025-09-29 104631" src="https://github.com/user-attachments/assets/3a5f72aa-abce-45e1-8a0d-572566289c9f" />



```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="847" height="688" alt="Screenshot 2025-09-29 104637" src="https://github.com/user-attachments/assets/a7e20399-377f-44ef-9a28-c340ef8fdcb2" />




```py
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
<img width="873" height="657" alt="Screenshot 2025-09-29 104644" src="https://github.com/user-attachments/assets/6059a8e2-6a84-4eee-80ef-654221334b86" />



```py
dt=pd.read_csv("titanic_dataset.csv")
dt
```

```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()
```

<img width="811" height="756" alt="Screenshot 2025-09-29 104659" src="https://github.com/user-attachments/assets/9ea5e3e2-be8d-4c54-a8a9-d71132ed7b7a" />


```py
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
<img width="878" height="630" alt="Screenshot 2025-09-29 104705" src="https://github.com/user-attachments/assets/5fb60fbe-13b1-48e4-b5ef-3a39927ec4d2" />




       
# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

       
