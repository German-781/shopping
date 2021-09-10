import csv
import sys
import random
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets

    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """

    evidencias = []
    etiquetas = []

    nombre_archivo = filename

    with open(nombre_archivo, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for r in reader:
            evidencia = []

            Administrative = int(r["Administrative"])
            evidencia.append(Administrative)

            Administrative_Duration = float(r["Administrative_Duration"])
            evidencia.append(Administrative_Duration)

            Informational = int(r["Informational"])
            evidencia.append(Informational)

            Informational_Duration = float(r["Informational_Duration"])
            evidencia.append(Informational_Duration)

            ProductRelated = int(r["ProductRelated"])
            evidencia.append(ProductRelated)

            ProductRelated_Duration = float(r["ProductRelated_Duration"])
            evidencia.append(ProductRelated_Duration)

            BounceRates = float(r["BounceRates"])
            evidencia.append(BounceRates)

            ExitRates = float(r["ExitRates"])
            evidencia.append(ExitRates)

            PageValues = float(r["PageValues"])
            evidencia.append(PageValues)

            SpecialDay = float(r["SpecialDay"])
            evidencia.append(SpecialDay)

            mes = r["Month"]
            if mes == 'Jan':
                Month = 0
            elif mes == 'Feb':
                Month = 1
            elif mes == 'Mar':
                Month = 2                
            elif mes == 'Apr':
                Month = 3
            elif mes == 'May':
                Month = 4
            elif mes == 'Jun':
                Month = 5
            elif mes == 'Jul':
                Month = 6
            elif mes == 'Aug':
                Month = 7            
            elif mes == 'Sep':
                Month = 8
            elif mes == 'Oct':
                Month = 9
            elif mes == 'Nov':
                Month = 10
            elif mes == 'Dec':
                Month = 11
            evidencia.append(Month)

            OperatingSystems = int(r["OperatingSystems"])
            evidencia.append(OperatingSystems)

            Browser = int(r["Browser"])
            evidencia.append(Browser)

            Region = int(r["Region"])
            evidencia.append(Region)

            TrafficType = int(r["TrafficType"])
            evidencia.append(TrafficType)

            Tipo_visitante = r["VisitorType"]
            if Tipo_visitante == "Returning_Visitor":
                VisitorType = 1
            else:
                VisitorType = 2

            evidencia.append(VisitorType)

            Finde = r["Weekend"]
            if Finde == "TRUE":
                Weekend = 1
            else:
                Weekend = 0
            evidencia.append(Weekend)

            Etiqueta = r["Revenue"]
            if Etiqueta == "TRUE":
                Revenue = 1
            else:
                Revenue = 0
            
            etiquetas.append(Revenue)
            evidencias.append(evidencia)

        return (evidencias,etiquetas)
            

    #raise NotImplementedError


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """

    model = KNeighborsClassifier(n_neighbors = 1)


    cercano = model.fit(evidence,labels)
    return cercano



    #raise NotImplementedError


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """


    largo_labels = len(labels)
    positivos = 0
    negativos = 0
    for label in labels:
        if label == 1:
            positivos += 1
        else:
            negativos += 1

    sensitivity = positivos / largo_labels
    specificity = negativos / largo_labels

    return sensitivity, specificity


    #raise NotImplementedError


if __name__ == "__main__":
    main()
