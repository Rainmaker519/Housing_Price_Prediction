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
#covariance = SUM((xi - avgi)(yj - avgj))/n.
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

#Creates a correlation matrix for the scalable variables, defined specifically in func.
#   Input: The dataframe.
#   Output: The correlation matrix as a dataframe.
def scalarCorrelationMatrix(data):
    scalar_columns = ['Posted On','BHK','Size','Bathroom','Floor On','Floor Out Of']

    data_columns = data.columns
    
    for i in scalar_columns:
        if not i in data.columns:
            scalar_columns.remove(i)
    
    correlation_columns = []

    for i in scalar_columns:
        list_column = []
        for j in scalar_columns:
            list_column.append(areScalarVariablesCorrelated(data,i,j))
        correlation_columns.append(pd.Series(list_column,name=i,dtype='float64'))

    correlation_frame = pd.concat(correlation_columns,axis=1)
    correlation_frame.index = scalar_columns
    
    return correlation_frame

#Scales 'Bathroom' to be total utility rooms (or 'BKH' + 'Bathroom') 
#divided by the square footage, then drops 'BHK'.
#   Input: Dataframe.
#   Output: Dataframe with scaled 'Bathroom' and dropped 'BHK'.
def sizeBathroomBHKScale(data):
    #First scale Bathroom and BHK by Size
    data.Bathroom = data.Bathroom.astype("float32",copy=True)
    data.BHK = data.BHK.astype("float32",copy=True)
    
    for i in range(len(data.Bathroom)):
        data.loc[data.Size == 0,'Size'] = 1
        data.Bathroom = data.Bathroom + data.BHK
        data.Bathroom = data.Bathroom / data.Size

    data = data.drop("BHK",axis=1)
    
    return data

#Returns the correlation between each of the categorical variables.
#   Input: Dataframe, and a list of strings representing the categorical variables if they differ from the usual needed.
#   Output: A list of lists with the format [crosstab,test_results,expected].
def getCatCorrelation(data,cat_vars = ["Area Type","Furnishing Status","Point of Contact","Bachelors","Family",
            "Bangalore","Chennai","Delhi","Hyderabad","Kolkata","Mumbai"]):
    result_holder = []
    for i in cat_vars:
        #cat_holder.append(data[i].value_counts())
        for j in cat_vars:
            crosstab, test_results, expected = rp.crosstab(data[i], data[j],
                                               test= "chi-square",
                                               expected_freqs= True,
                                               prop= "cell")
            result_holder.append([crosstab,test_results,expected])
        
    return result_holder

#Returns a correlation matrix given the results of getCatCorrelation,
#and contains correlation strength based on cutoffs for coloring.
#   Input: Results of getCatCorrelation().
#   Output: Correlation matrix.
def catCorrMatrix(catCorrResults):
    corr_strength = pd.DataFrame()
    corr_strength_constructor = []

    #n is the starting point (index wise) for the current var
    n = 0
    interp = False
    for i in range(len(cat_vars)): 
        corr_strength_helper = []
        for j in range(len(cat_vars)):
            if interp:
                if abs(result[n+j][1]['results'][2]) > .25:
                    #print("Very Strong")
                    corr_strength_helper.append('VS')
                elif abs(result[n+j][1]['results'][2]) > .15:
                    #print("Strong")
                    corr_strength_helper.append('S')
                elif abs(result[n+j][1]['results'][2]) > .1:
                    #print("Moderate")
                    corr_strength_helper.append('M')
                elif abs(result[n+j][1]['results'][2]) > .05:
                    #print("Weak")
                    corr_strength_helper.append('W')
                else:
                    #print("None or Very Weak")
                    corr_strength_helper.append('N')
            else:
                corr_strength_helper.append(result[n+j][1]['results'][2])

        corr_strength_helper = pd.Series(corr_strength_helper,name=cat_vars[i])
        corr_strength = pd.concat([corr_strength,corr_strength_helper],axis=1)

        n = n + len(cat_vars)

    corr_strength.index = cat_vars
    
    return corr_strength

#Function to be used by dataframe.style.apply() for coloring frame.
#   Input: Correlation matrix for cat variables.
#   Output: Boolean based on what conditions were satisfied.
def highlightS(x,color):
    ones = np.where(x > .99, "color: white;", None)
    s = np.where(x > .2, f"color: {color};", None)
    vs = np.where(x > .5, "color: blue;", None)
    for i in range(len(vs)):
        if ones[i] == None:
            if not s[i] == None:
                if vs[i] == None:
                    vs[i] = s[i]
        else:
            vs[i] = ones[i]
    return vs

#Function to be used by dataframe.style.apply() for hiding uninformative parts of frame.
#   Input: Correlation matrix for cat variables.
#   Output: Boolean based on what conditions were satisfied.
def highlightN(x,color):
    return np.where((x == "N"), f"color: {color};", None)

#display(corr_strength.style.apply(highlightS,color="green"))

#Combines all heavily collinear cities into a single column 'InSimilarCities'.
#   Input: Dataframe.
#   Output: Refactored dataframe.
def consolidateCitiesIntoSimilarEffect(data):
    for i in ["Bangalore","Chennai","Hyderabad","Delhi"]:
        data.loc[data[i] == 1,'Delhi'] = 1
        
    data = data.drop("Bangalore",axis=1)
    data = data.drop("Chennai",axis=1)
    data = data.drop("Hyderabad",axis=1)
    data = data.drop("Kolkata",axis=1)
    
    data = data.rename(columns={"Delhi":"InSimilarCities"})
    
    return data

#Removes data points over two standard deviations away from the mean of a continous variable.
#   Input: Dataframe, and column to prune unless column is 'Rent'.
#   Output: Adjusted dataframe.
def removeContinuousOutliers(column="Rent"):
    col = column
    sample_mean = np.mean(data[col],axis=0)
    sample_std_dev = np.std(data[col],axis=0)
    row_pointer = 0
    for row in range(len(data)):
        safe = True
        val = data.iloc[row_pointer][col]
        if val <= sample_mean - 2 * sample_std_dev:
            safe = False
        elif val >= sample_mean + 2 * sample_std_dev:
            safe = False

        if not safe:
            data = data.drop(row,axis=0)
            row_pointer = row_pointer - 1

        row_pointer = row_pointer + 1
    data.reset_index(drop=True)

    return data