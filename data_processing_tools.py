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

#Checks whether a linear relationship exists between the provided column and 'Rent'.
#   Input: The name of the column to compare against 'Rent' as a string.
#   Output: True if doesn't fail, displays a graph of the relationship.
def check_linear_relationship(compare_from):
    compare_col = compare_from
    theta = np.polyfit(data[compare_col], data['Rent'],1)
    y_line = theta[1] + theta[0] * data[compare_col]
    plt.scatter(data[compare_col], data['Rent'], color='red')
    plt.plot(data[compare_col], y_line, 'b')
    plt.title('Rent Vs ' + str(compare_col), fontsize=14)
    plt.xlabel(compare_col, fontsize=14)
    plt.ylabel('Rent', fontsize=14)
    plt.grid(True)
    plt.show()
    print("The slope of the best fit for", compare_col, "is " + str(theta[0]))

    return True

#Shows how the MFE changes as each column is added until all columns are used.
#PCA used each time to mitigate collinearity and allow the number of features to be consistent over tests.
#   Input: A pandas dataframe, the type of model/regression to use, and the number of repeats to reduce potential outlier runs.
#   Output: True given no errors thrown, and shows the pyplot of the results.
def validate(data,reg_type="linear_regression",repeats=3):
    columns = data.columns
    X = data[columns]
    y = np.ravel(data[["Rent"]])
    X = X.drop("Rent",axis=1)
    num_points = len(data)
    
    pca = PCA()
    X_reduced = pca.fit_transform(scale(X))
    
    cv = RepeatedKFold(n_splits=10, n_repeats=repeats, random_state=1)
    
    if reg_type == 'linear':
        regr = LinearRegression()
    elif reg_type == 'random_forest':
        regr = RandomForestRegressor()
    elif reg_type == 'lasso':
        regr = Lasso()
    else:
        print("you need a valid regression type for pca")
        return
    mse = []
    
    score = -1*model_selection.cross_val_score(regr,
           np.ones((len(X_reduced),1)), y, cv=cv,
           scoring='neg_mean_squared_error').mean()  
    mse.append(score/num_points)
    
    for i in np.arange(1, len(data.columns)):
        score = -1*model_selection.cross_val_score(regr,
               X_reduced[:,:i], y, cv=cv, scoring='neg_mean_squared_error').mean()
        mse.append(score/num_points)
        
    plt.plot(mse)
    plt.xlabel('Number of Used Components')
    plt.ylabel('Total MSE / Number of Rows')
    plt.title('Rent')
    display(mse)
    
    return True

#Gives the residuals for a regression as a list.
#   Input: The dataframe, a sklearn model, and a boolean representing whether or not the model has been trained.
#   Output: A list of residuals from the regression's predicted rent to the actual rent.
def get_residuals(data,skl_model,trained):
    y = data["Rent"]
    X = data.drop("Rent",axis=1)
    X_train = X[:int(len(X)*.7)]
    y_train = y[:int(len(X)*.7)]
    X_test = X[int(len(X)*.7):]
    y_test = y[int(len(X)*.7):]
    
    
    X_train = X_train.reset_index()
    X_test = X_test.reset_index()
    y_train = y_train.reset_index()
    y_test = y_test.reset_index()
    
    
    if not trained:
        skl_model.fit(X_train,y_train)
    
    full_predictions = skl_model.predict(X_test)
    predictions = []
    residuals = []
    
    y_test = y_test.drop("index",axis=1)
    
    for i in range(len(full_predictions)):
        predictions.append(full_predictions[i][1])
    
    for i in range(len(predictions)):
        residuals.append(abs(predictions[i] - y_test["Rent"][i]))
           
    return residuals

#Tests for correlation between two scalar variables.
#covariance = SUM((xi - avgi)(yj - avgj))/n
#   Input: The dataframe, the name of the first column, the name of the column to compare against the first.
#   Output: The covariance of the two given columns.
def areScalarVariablesCorrelated(data,varA,varB):
    var1 = scale(data[varA])
    var2 = scale(data[varB])
    
    avg1 = var1.mean()
    avg2 = var2.mean()
    
    if len(var1) != len(var2):
        print("Please use variables with the same number of entries.")
        return None
    
    total_sum = 0
    for i in range(len(var1)):
        total_sum = total_sum + (var1[i] - avg1)*(var2[i] - avg2)
    
    return total_sum/len(var1)