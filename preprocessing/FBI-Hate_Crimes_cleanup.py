import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer

df=pd.read_csv("hate_crime.csv")

#drop unused columns
unused = ['TOTAL_INDIVIDUAL_VICTIMS','ORI','PUB_AGENCY_UNIT','STATE_NAME', 'DIVISION_NAME','POPULATION_GROUP_CODE','ADULT_VICTIM_COUNT', 'JUVENILE_VICTIM_COUNT','ADULT_OFFENDER_COUNT','JUVENILE_OFFENDER_COUNT','OFFENDER_ETHNICITY','MULTIPLE_OFFENSE','MULTIPLE_BIAS']
df.drop(columns=unused,inplace=True)

#set numeric columns to proper data type
int64 = ['INCIDENT_ID', 'DATA_YEAR', 'TOTAL_OFFENDER_COUNT', 'VICTIM_COUNT']
df['INCIDENT_DATE'] = pd.to_datetime(df['INCIDENT_DATE'])

for num in df:
    if num in int64:
        df[num] = df[num].astype(np.int64)

#only get cities
df = df.loc[df.AGENCY_TYPE_NAME == 'City']

#OHE for cat features
cat_encoder = OneHotEncoder()

#define features for OHE:
k = ['OFFENDER_RACE']

for f in k:
    transformed = cat_encoder.fit_transform(df[[f]])
    temp_df = pd.DataFrame(transformed.toarray(),columns = cat_encoder.get_feature_names_out())
    df = pd.concat([df.reset_index(drop=True), temp_df.reset_index(drop=True)], axis=1)
    df.drop(columns=f,inplace=True)

#define features for multilable binarizer
mlb = MultiLabelBinarizer()

b = ['OFFENSE_NAME','VICTIM_TYPES','BIAS_DESC','LOCATION_NAME']
for l in b:
    df[l] = [set(lbl.split(";")) for lbl in df[l]]
    mlb.fit_transform(df[l].tolist())
    mb = pd.DataFrame(mlb.fit_transform(df[l].tolist()),columns=np.char.add(l +"_", mlb.classes_))
    df = pd.concat([df, mb], axis=1)
    df.drop(columns=l,inplace=True)