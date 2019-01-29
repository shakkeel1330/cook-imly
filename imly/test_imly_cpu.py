import automation_script
import pandas as pd
import numpy as np
from os import path

dataset_name = "uci_iris" # Name of your dataset as mentioned in the 
dataset_info = automation_script.get_dataset_info(dataset_name)

# Gathering data and converting it into a dataframe
url = dataset_info['url']
data = pd.read_csv(url , delimiter=",", header=None, index_col=False)

# This part of the preparation is specific to the dataset
class_name,index = np.unique(data.iloc[:,-1],return_inverse=True)
data.iloc[:,-1] = index
data = data.loc[data[4] != 2]
X = data.iloc[:,:-1]
Y = data.iloc[:,-1]

automation_script.run_imly(dataset_info=dataset_info, 
                                      model_name='logistic_regression', 
                                      X=X, Y=Y, 
                                      test_size=0.60)