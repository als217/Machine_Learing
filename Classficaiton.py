
#Load libs
import getopt
import time
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from csv import reader
import sys
from tabulate import tabulate

# Extract data function
def banner():
    print("""\033[38;5;38m  
      ______  _                     _   ___  _                     _               
     / _____)| |                   (_) / __)(_)              _    (_)              
    | /      | |  ____   ___   ___  _ | |__  _   ____  ____ | |_   _   ___   ____  
    | |      | | / _  | /___) /___)| ||  __)| | / ___)/ _  ||  _) | | / _ \ |  _ \ 
    | \_____ | |( ( | ||___ ||___ || || |   | |( (___( ( | || |__ | || |_| || | | |
     \______)|_| \_||_|(___/ (___/ |_||_|   |_| \____)\_||_| \___)|_| \___/ |_| |_|
                                                                                    \033[0;0m""")
    print("""\033[38;5;38m
    - Classification Tools
    - Made By: Abdullah Alharbi   | 25 Mar 2022
    - Supervise By : Dr.Ghassan Bate
    - Thanks to : https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
    - Version : 2.0 \033[0;0m""",end="\n\n\n")

def extract_data(filename):
    with open(filename, 'r') as DataSet_file:
        dataset = reader(DataSet_file)
        headers = next(dataset)
        length_headers = len(headers)
        dataset = read_csv(DataSet_file, header=0, names=headers, na_values='.')
        return dataset , length_headers , headers

# show graph of data
def info_about_data(dataset,headers):
    #shape
    print("\033[38;5;40m Size of dataset : \033[0;0m \n{} ".format(dataset.shape),end="\n\n")
    #head
    print("\033[38;5;40m head of dataset : \033[0;0m \n{} ".format(dataset.head(20)),end="\n\n")
    #descriptions
    print("\033[38;5;40m Describe dataset : \033[0;0m \n{}".format(dataset.describe()), end="\n\n")
    #Target distribution
    print("\033[38;5;40m Size of target : \033[0;0m \n{}".format(dataset.groupby(headers[-1]).size()), end="\n\n")

    # box and whisker plots
    dataset.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False)
    #histgram
    dataset.hist()
    #Scatter plot matrix
    scatter_matrix(dataset)
    #pyplot.show()                  --> uncomment this line if you want to show all graph

def train_test_data(dataset, length_Of_header, Target_Position):
    # Split-out validation dataset
    array = dataset.values
    X = array[:,0:length_Of_header -1]
    y = array[:,Target_Position -1]
    X_train, X_validation,Y_train,Y_validation = train_test_split(X,y,test_size=0.20, random_state=1)
    return X_train, Y_train, X_validation, Y_validation

# algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
def Check_Algorithms(models,X_train,Y_train):
    # Spot Check Algorithms
    results_mean = []
    results_std = []
    names = []

    for name, model in models:
        kfold = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
        cv_results = cross_val_score(model,X_train,Y_train,cv=kfold,scoring='accuracy')
        results_mean.append(cv_results.mean())
        results_std.append(cv_results.std())
        names.append(name)
    results_max = max(results_mean)
    results_max = results_mean.index(results_max)
    Best_models = models[results_max]
    return names, results_mean, results_std, Best_models

#Predictions
def Predictions(Best_Module,X_train,Y_train,X_Validation,Y_Validation):
    #print(Best_Module)
    model = Best_Module[1]
    model.fit(X_train,Y_train)
    predictions = model.predict(X_Validation)
    AUC_score = accuracy_score(Y_Validation,predictions)
    CM_score = confusion_matrix(Y_Validation,predictions)
    CR_score = classification_report(Y_Validation,predictions)
    return AUC_score,CM_score,CR_score
# Print results in table
def show_results(names, results_mean,result_std,AUC_score,CM_score,CR_score):
    headers = ['Model Name', 'Accuracy Mean','Accuracy STD']
    table = zip(names, results_mean, result_std)
    print("\033[38;5;21m Results : \033[0;0m")
    print(tabulate(table,headers=headers, tablefmt='pipe', stralign='center'),)
    print("\n\033[38;5;21m The Best Accuracy is :\033[0;0m {}".format(AUC_score))
    print("\033[38;5;21m The Confusion_matrix :\033[0;0m \n{}".format(CM_score))
    print("\033[38;5;21m The Classification_report :\033[0;0m \n{}".format(CR_score))

def main():
    # Check if there are argument or exit the program
    args = sys.argv[1:]
    if not args:
        print("""
        if you have some issues about running this tools,
        could you to check these things in your dataset:
            1- check the target have to be in last column.
            2- the dataset have to be in csv.
            3- teh dataset have to be spread with ','.
            4- install all Packages (scipy, numpy, matplotlib, pandas, sklearn)  
            if you have something else contact me in github.
            
        \033[38;5;21m[Usage]:\033[0;0m [-f | --filename] file [file ...]         upload dataset
        """)
        sys.exit(1)
    #defint filename
    filename = ''
    try :
        opts, args = getopt.getopt(args,"hf:",["filename="])
    except getopt.GetoptError:
        print ('test.py -i <inputFile>')
        sys.exit(2)

    for opt, arg in opts:       # using for loop, if we want to add new arguments
        if opt == '-h':
            print('\033[38;5;21m[Usage]:\033[0;0m Classficaiton_Breastedit.py -i <inputFile> ' )
            sys.exit()
        elif opt in ("-f", "--filename"):
            filename = arg
    # Run all function
    banner()
    time.sleep(1.5)
    dataset , length_header , headers = extract_data(filename)
    info_about_data(dataset, headers)
    target = length_header
    X_train , Y_train, X_Validation, Y_validation= train_test_data(dataset,length_header,target)
    names, results_mean, results_std, Best_models = Check_Algorithms(models,X_train,Y_train)
    AUC_score, CM_score, CR_score = Predictions(Best_models,X_train,Y_train,X_Validation,Y_validation)
    show_results(names, results_mean, results_std,AUC_score,CM_score,CR_score)

if __name__ == '__main__':
    main()
