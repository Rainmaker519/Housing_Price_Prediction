import numpy as np
import pandas as pd

from pandas.api.types import CategoricalDtype

from sklearn.preprocessing import OneHotEncoder

#Helper function for dateSplit.
#   Input: A list of strings in the form "2022-01-01".
#   Output: A list of lists in the form [2022, 01, 01].
def postedtoscalar_dateSplit(dates):
    split_dates = []

    for i in range(len(dates)):
        split_dates.append(dates[i].split("-"))
    
        if split_dates[i][0] != "2022":
            return False
        
    return split_dates

#Helper function for dateSplit.
#   Input: A month and a year.
#   Output: The number of days in that month taking leap years into account.
def postedtoscalar_numberOfDays(y, m):
      leap = 0
      if y% 400 == 0:
         leap = 1
      elif y % 100 == 0:
         leap = 0
      elif y% 4 == 0:
         leap = 1
      if m==2:
         return 28 + leap
      list = [1,3,5,7,8,10,12]
      if m in list:
         return 31
      return 30

#This function is used to convert 'Floor' into two columns 'Floor On' and 'Floor Out Of'
#   Input: A dataframe
#   Output: The dataframe with the 'Floor' column replaced with 'Floor On' and 'Floor Out Of'.
def splitFloorIntoTwo(data):
    original_floor = data["Floor"]
    floor_on = []
    floor_out_of = []
    for i in range(len(original_floor)):
        split = original_floor[i].split(" out of ")
        if len(split) == 1:
            if split[0] == 'Ground':
                floor_on.append(0)
                floor_out_of.append(0)
            else:
                floor_on.append(split[0])
                floor_out_of.append(split[0])
        else:  
            #Ground is 0, others are basement and such so all other non-int convertables go to -1
            if split[0] == 'Ground':
                floor_on.append(0)
            else:
                try:
                    floor_on.append(int(split[0]))
                except:
                    floor_on.append(-1)
            floor_out_of.append(int(split[1]))
    
    floor_on = np.array(floor_on)
    floor_out_of = np.array(floor_out_of)
    
    data["Floor On"] = floor_on
    data["Floor Out Of"] = floor_out_of

    data.drop("Floor",axis=1,inplace=True)

    return data

#This function is used to label encode the categorical data. 
#(Treats 'Furnishing Status' differently as the categories are ordered.)
#   Input: A dataframe and the label of the column to be label encoded as input.
#   Output: The dataframe with the column label encoded.
def labelEncodeColumn(data,column_name):
    if column_name == "Furnishing Status":
        cat_type = CategoricalDtype(categories=["Unfurnished", "Semi-Furnished", "Furnished"], ordered=True)
        data[column_name] = data[column_name].astype(cat_type)
        data[column_name] = data[column_name].cat.codes
    else:
        data[column_name] = data[column_name].astype('category')
        data[column_name] = data[column_name].cat.codes

    return data

#This function is used to one hot encode categorical data specifically for this dataset.
#   Input: A dataframe and the label of the column to be one hot encoded as input.
#   Output: The dataframe with the column one hot encoded.
def oneHotEncodeColumn(data,column_name):
    encoder = OneHotEncoder(handle_unknown='ignore',sparse=False)
    if column_name == "Tenant Preferred":
        bachelors = []
        family = []
        for i in range(len(data["Tenant Preferred"])):
            if data["Tenant Preferred"][i] == "Bachelors":
                bachelors.append(1)
                family.append(0)
            elif data["Tenant Preferred"][i] == "Family":
                bachelors.append(0)
                family.append(1)
            elif data["Tenant Preferred"][i] == "Bachelors/Family":
                bachelors.append(1)
                family.append(1)
            else:
                print("issue w tenant preferred encoding")
        bachelors = pd.Series(bachelors)
        bachelors.name = "Bachelors"
        family = pd.Series(family)
        family.name = "Family"

        data = pd.concat([data,bachelors,family],axis=1)

        return data

    else:
        encoded_data = pd.DataFrame(encoder.fit_transform(data[[column_name]]))

        feature_names = encoder.get_feature_names_out()
        for i in range(len(feature_names)):
            feature_names[i] = feature_names[i].split("_")
        for i in range(len(feature_names)):
            feature_names[i] = feature_names[i][1]

        encoded_data.columns = feature_names

        data = pd.concat([data,encoded_data],axis=1)

    return data

#This function is used to drop all rows with 'Built Area' as the 'Area Type'.
#   Input: A dataframe.
#   Output: The dataframe with all rows with 'Built Area' as the 'Area Type' dropped, and the index reset.
def drop_built_area(data):
    data = data.drop(data[data['Area Type'] == 'Built Area'].index)
    data = data.reset_index()

    return data

### STOPPED MOVING OVER BEFORE FUNCTIONS FOR CHECKING LINEAR RELATIONSHIPS ###