# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "feb712bf-1546-445c-bea4-665c82c12f2b",
# META       "default_lakehouse_name": "pbisummitlake",
# META       "default_lakehouse_workspace_id": "4a689fc0-cc6e-482f-b6cf-3026a575a7c1"
# META     },
# META     "environment": {
# META       "environmentId": "28dbaf6b-8fc5-89a4-444a-db3b58a22f66",
# META       "workspaceId": "00000000-0000-0000-0000-000000000000"
# META     }
# META   }
# META }

# CELL ********************

import pandas as pd
import random

# Define possible values for each feature
household_keys = [f'HH_{i}' for i in range(1, 101)]
age_desc = ['19-24', '25-34', '35-44', '45-54', '55-64', '65+']
marital_status_code = ['A', 'B', 'C', 'U']
income_desc = ['Under 15K', '15-24K', '25-34K', '35-49K', '50-74K', '75-99K', '100-124K', '125K+']
homeowner_desc = ['Homeowner', 'Renter']
hh_comp_desc = ['Single Male', 'Single Female', '2 Adults No Kids', '2 Adults Kids', 'Single Parent', 'Other']
household_size_desc = ['1', '2', '3', '4+']
kid_category_desc = ['None', '1', '2', '3+']

# Generate 100 rows of data
data = {
    'household_key': random.choices(household_keys, k=100),
    'AGE_DESC': random.choices(age_desc, k=100),
    'MARITAL_STATUS_CODE': random.choices(marital_status_code, k=100),
    'INCOME_DESC': random.choices(income_desc, k=100),
    'HOMEOWNER_DESC': random.choices(homeowner_desc, k=100),
    'HH_COMP_DESC': random.choices(hh_comp_desc, k=100),
    'HOUSEHOLD_SIZE_DESC': random.choices(household_size_desc, k=100),
    'KID_CATEGORY_DESC': random.choices(kid_category_desc, k=100),
    'Churn': random.choices([0, 1], k=100)  # Binary label for churn (0 or 1)
}

# Create DataFrame
df_pandas = pd.DataFrame(data)

df = spark.createDataFrame(df_pandas)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import mlflow
from synapse.ml.predict import MLFlowTransformer
    

    
model = MLFlowTransformer(
        inputCols=['household_key','AGE_DESC' , 'MARITAL_STATUS_CODE' , 'INCOME_DESC' , 'HOMEOWNER_DESC' , 'HH_COMP_DESC' , 'HOUSEHOLD_SIZE_DESC' , 'KID_CATEGORY_DESC'], # Your input columns here
        outputCol="predictions", # Your new column name here
        modelName="demo-churn-dunhumby", # Your model name here
        modelVersion=2 # Your model version here
    )
df_transform = model.transform(df)
    
#df_transform.write.format('delta').mode("overwrite").save('predictions') # Your output table filepath here


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df_transform.printSchema()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
