#done without numpy
from random import seed
from random import randrange
from csv import reader
from math import sqrt

# Load a CSV file
def load_csv(filename):
    """
    Description
    ---
    Loads a CSV file from disk as a csv.reader object  and
    outputs a python list object. the first column should be the independent variable
    and the second column would be the dependent variable

    Parameters
    ---
    filename :  file object
                CSV file that contains the values for the dependent variable 'Y'
                and the independent variable 'X'. using '\t' delimiter by default
                and non european decimal (. and not , for marking decimals in
                floats)

    Returns
    ---
    dataset :   list
                list of lists. the dataset list object is made up of each
                individual data element from the file as a list itself
                e.g. [[a,b], [c,d],[e,f]...]
                the variables a, b, c...etc are floats

    Raises
    ---
    no error handling implemented but could add error for particular IO errors


    """
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file, delimiter='\t')
        for row in csv_reader:
            if not row:
                continue
            dataset.append([float(i) for i in row])
    return dataset


# Split a dataset into a train and test set
def train_test_split(dataset, split):
    """
    Description 
    ---
    splits the incoming dataset list into two distinct list for training and testing
    it creates two list by randomly  popping elements from a dataset list copy and 
    appending it to the training dataset list
    
    Parameters
    ---
    dataset: list
             list made from the read csv file probably dataset :   list
                list of lists. the dataset list object is made up of each
                individual data element from the file as a list itself
                e.g. [[a,b], [c,d],[e,f]...]
                the variables a, b, c...etc are floats
    split:   float 
             split < 1; percentage value that will dictate the train/test 
             split of the dataset list
    
    Returns
    ---
    train:   list
             a list object constructed from popped elements of a dataset list object in the format of 
             [[a,b],[c,d]...] where a, b, c, etc are float values
    
    (test) dataset_copy: list
                  a list object that is what is remained after its elements have been popped 
                  appended to the train list object

    Raises: nothing
             
    """
    train = list()
    train_size = split * len(dataset)
    dataset_copy = list(dataset) 
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        train.append(dataset_copy.pop(index))
    return train, dataset_copy

# Calculate root mean squared error
def rmse_metric(actual, predicted):
    """
    Description
    ---
    calculate the root mean square error 

    Parameters
    ---
    actual:     list
                a list containing actual float values 
    
    predicted   list
                a list object containing predicted values as floats

    Returns:    
    rmse:       float
                the square root of the mean_error 

    Raises: Nothing
            
    """
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)

# Evaluate an algorithm using a train/test split
def evaluate_algorithm(dataset, algorithm, split, *args):
    """
    Description
    ---
    this evaluates the accuracy of the algorithm by returning  the root mean square erro.  
    the numerical calculation is done by the rmse_metric function and this function does the
    actual set up stuff such as running the algorithm and getting the actual values ready 
    so that they can both be passed to rmse_metric

    Parameters
    ---
    dataset :   list
                list of lists. the dataset list object is made up of each
                individual data element from the file as a list itself
                e.g. [[a,b], [c,d],[e,f]...]
                the variables a, b, c...etc are floats

    algorithm: function (returns: list)
               in this case the algorithm will be simple_linear_regression function 
               and will return a list object.

    split:     float
               split < 1; percentage value that will dictate the train/test 
               split of the dataset list
    
    Returns
    ---
    rmse:      float
               root mean square error

    Raises
    ---
    fully fleshed out this shouldnt raise any errors because it calls upon elements that should already have been
    sanitized or whatever.
               
    
    """
    train, test = train_test_split(dataset, split)
    test_set = list()
    for row in test:
        """
        this loop takes a previous split dataset test list object and removes the last
        element from each row which is the 'y' or dependent variable.
        
        """
        row_copy = list(row)
        row_copy[-1] = None
        test_set.append(row_copy)
    predicted = algorithm(train, test_set, *args)
    actual = [row[-1] for row in test]
    rmse = rmse_metric(actual, predicted)
    return rmse

# Calculate the mean value of a list of numbers
def mean(values):
    """
    Description
    ---
    calculate the mean value of a list object containing numbers

    Parameters:
    ---
    values: list
            a list object consisting of float values
    
    Returns
    ---
    mean: float
          the mean 
    
    """
    return sum(values) / float(len(values))

# Calculate covariance between x and y
def covariance(x, mean_x, y, mean_y):
    """
    Description
    ---
    calculates the covariance value between x and y

    Parameters
    ---
    x:    list
    a list object containing the INDEPENDENT variables of the dataset
    
    y:    list
    a list object containing the DEPENDENT variables of the dataset
    
    mean_x:    float
    mean of the INDEPENDENT variable x as a float value

    mean_y:    float
    mean of the dependent variable y ass a float value

    Returns:
    ---
    covar:    float
    the covariance value as a float
    """
    
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - mean_x) * (y[i] - mean_y)
    return covar

# Calculate the variance of a list of numbers
def variance(values, mean):
    """
    Description
    ---
    calculate variance 
    
    Parameters
    ---
    values:    list
    a list of floats

    mean:    float
    the mean

    Returns
    ---
    variance:    float
    you know
    """
    return sum([(x-mean)**2 for x in values])

# Calculate coefficients
def coefficients(dataset):
    """
    Description
    ---
    the simple linear regression is defined to be linear by its coefficients
    Bo, B1. we calculate those here. it's based on a derived equation after one does 
    calculus to find minimum value etc etc. the equations can be found online or derived 
    by hand

    Parameters
    ---
    dataset :   list
                list of lists. the dataset list object is made up of each
                individual data element from the file as a list itself
                e.g. [[a,b], [c,d],[e,f]...]
                the variables a, b, c...etc are floats

    Returns
    ---
    [b0,b1]:    list 
                a list object containing the calculated coefficients for the simple 
                linear regression model
    """
    x = [row[0] for row in dataset]
    y = [row[1] for row in dataset]
    x_mean, y_mean = mean(x), mean(y)
    b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
    b0 = y_mean - b1 * x_mean
    return [b0, b1]

# Simple linear regression algorithm
def simple_linear_regression(train, test):
    """
    Description
    ---
    this function is effectively the top of the calculation "pyramid" so to speak
    this is the model and all of it's necessary calculations are done by the other 
    functions. covariance, mean, coefficients, etc.

    Parameters
    ---
    train:    list
              some list object with the training data in the format of [[a,b],[c,d]...] where
              a, b, c, etc are float values
    test:     list
              some list object with the testing  data in the format of [[a,b],[c,d]...] where
              a, b, c, etc are float values
    """
    predictions = list()
    b0, b1 = coefficients(train)
    for row in test:
        yhat = b0 + b1 * row[0]
        predictions.append(yhat)
    return predictions

# Simple linear regression on insurance dataset
seed(1)
# load and prepare data
filename = 'testdata.csv'
dataset = load_csv(filename)


# evaluate algorithm
split = 0.6
rmse = evaluate_algorithm(dataset, simple_linear_regression, split)
print('RMSE: %.3f' % (rmse))
