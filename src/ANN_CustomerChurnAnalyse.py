import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Aufgabe: Finde Kunden die höchstwahrscheinlich kündigen werden oder ob neue Kunden kündigen werden oder nicht
# Customer Churn Neuronal Network

# Python 3.9 venv !!!
# pip install "tensorflow<2.11"
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117 --upgrade --force-reinstall

# Data processing ------------------------------------------------------------------------
# DS: Course-Codes-Datasets/Part 8 - Deep Learning/Section 39 - Artificial Neural Networks (ANN)/Python/Churn_Modelling.csv
# Spalte "exited": 0 - noch Kunde; 1 - kein Kunde mehr

if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


    print(f'Part1 ANN Training with bank data...')

    # Importing the dataset
    dataset = pd.read_csv('src/data/Churn_Modelling.csv')
    X = dataset.iloc[:, 3:-1].values
    y = dataset.iloc[:, -1].values
    # print(X)
    # print(y)

    # Encoding categorical data
    # Label Encoding the "Gender" column
    le = LabelEncoder()
    X[:, 2] = le.fit_transform(X[:, 2])
    # print(X)

    # One Hot Encoding the "Geography" column
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
    X = np.array(ct.fit_transform(X))
    print(X)

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Feature Scaling  --- EXTREM WICHTIG
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)


    # Define the ANN ------------------------------------------------------------------------
    # Initializing the ANN
    with tf.device('/CPU:0'):
        ann = tf.keras.models.Sequential()
        # Adding the input layer and the first hidden layer
        ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
        # Adding the second hidden layer
        ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
        # Adding the output layer
        ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


        # Train the ANN ------------------------------------------------------------------------
        # Compiling the ANN
        ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        # Training the ANN on the Training set
        ann.fit(X_train, y_train, batch_size=32, epochs=100)

        """
        Use our ANN model to predict if the customer with the following informations will leave the bank: 
        Geography: France
        Credit Score: 600
        Gender: Male
        Age: 40 years old
        Tenure: 3 years
        Balance: $ 60000
        Number of Products: 2
        Does this customer have a credit card? Yes
        Is this customer an Active Member: Yes
        Estimated Salary: $ 50000
        So, should we say goodbye to that customer?
    
        Solution:
        """
        print("Kunde kündigen? ", ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)


    # Predicting the Test set results
    y_pred = ann.predict(X_test)
    y_pred = (y_pred > 0.5)
    print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix, accuracy_score
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix: ", cm)
    print("Genauigkeit: ", accuracy_score(y_test, y_pred))

