from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
from django.conf import settings
import os
from django.core.files.storage import FileSystemStorage
import os
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("Dataset/Customer_Data.csv")
dataset.drop(['Customer_ID'], axis = 1,inplace=True)
labels = np.unique(dataset['Customer_Status'])
#Pre-process the data by converting categorical variables to numerical variables, and replacing missing values with mean.
label_encoder = []
columns = dataset.columns
types = dataset.dtypes.values
for i in range(len(types)):
    name = types[i]
    if name == 'object': #finding column with object type
        le = LabelEncoder()
        dataset[columns[i]] = pd.Series(le.fit_transform(dataset[columns[i]].astype(str)))#encode all str columns to numeric
        label_encoder.append([columns[i], le])
dataset.fillna(dataset.mean(), inplace = True)
#dataset normalizing using standard Scaler
Y = dataset['Customer_Status'].ravel()
dataset.drop(['Customer_Status'], axis = 1,inplace=True)
X = dataset.values
sc = StandardScaler()
X = sc.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
rf_cls = RandomForestClassifier()
rf_cls.fit(X_train, y_train)

def Predict(request):
    if request.method == 'GET':
        return render(request, 'Predict.html', {})


def ViewFile(request):
    if request.method == 'GET':
        return render(request, 'ChurnPrediction.html', {})

def PredictAction(request):
    if request.method == 'POST':
        global dataset, rf_cls, label_encoder, sc, labels
        myfile = request.FILES['t1'].read()
        fname = request.FILES['t1'].name
        if os.path.exists("ChurnApp/static/"+fname):
            os.remove("ChurnApp/static/"+fname)
        with open("ChurnApp/static/"+fname, "wb") as file:
            file.write(myfile)
        file.close()

        testdata = pd.read_csv("ChurnApp/static/"+fname)
        temp = testdata.values
        for i in range(len(label_encoder)):
            le = label_encoder[i]
            if le[0] != 'Customer_Status':
                testdata[le[0]] = pd.Series(le[1].transform(testdata[le[0]].astype(str)))  # encode all str columns to numeric                
        testdata.fillna(dataset.mean(), inplace=True)  # replace missing values        
        testdata = testdata.values
        testdata = sc.transform(testdata)
        predict = rf_cls.predict(testdata)
        predict = predict.ravel()

        label_map = {
                        "Churned": "Likely to churn",
                        "Joined": "Just joined",
                        "Stayed": "Likely to Stay"
                    }

        # Prepare data for rendering
        table_data = []
        for i in range(len(predict)):
            original_label = labels[predict[i]]
            display_label = label_map.get(original_label, original_label)  # fallback if not found
            row_data = {
                            'test_data': temp[i],
                            'prediction': display_label,
                            'color_class': 'text-danger' if predict[i] == 0 else ('text-primary' if predict[i] == 1 else 'text-success')
                        }

            table_data.append(row_data)


        # Pass data to context
        context = {'table_data': table_data}
        return render(request, 'Graph.html', context)   

def Visualization(request):
    if request.method == 'GET':
        return render(request, 'Visualization.html', {})

def VisualizationAction(request):
    if request.method == 'POST':
        column = request.POST.get('t1', False)
        data = pd.read_csv("Dataset/Customer_Data.csv")
        gender_churn = data.groupby([column, 'Customer_Status']).size().reset_index()
        gender_churn = gender_churn.rename(columns={0: 'Count'})

        if column == 'Tenure_in_Months':
            min_tenure = data['Tenure_in_Months'].min()
            max_tenure = data['Tenure_in_Months'].max()
            bin_edges = list(range((min_tenure // 10) * 10, ((max_tenure // 10) + 2) * 10, 10))
            bin_labels = [f'{bin_edges[i]}-{bin_edges[i + 1]}' for i in range(len(bin_edges) - 1)]
            data['Tenure Range'] = pd.cut(data['Tenure_in_Months'], bins=bin_edges, labels=bin_labels, right=False)
            grouped_dynamic = data.groupby(['Tenure Range', 'Customer_Status']).size().reset_index(name='Count')
            plt.figure(figsize=(12, 6))
            sns.barplot(data=grouped_dynamic, x='Tenure Range', y='Count', hue='Customer_Status')
            plt.title('Customer Status by Dynamic Tenure Range')
            plt.xlabel('Tenure in Months Range')
            plt.ylabel('Number of Customers')
            plt.xticks(rotation=45)
            plt.legend(title='Customer Status')

        elif column == 'State':
            sns.set(style="whitegrid", palette="muted")
            plt.figure(figsize=(16, 8))
            sns.countplot(data=data, x='State', hue='Customer_Status')
            plt.title('Customer Status Distribution by State', fontsize=16)
            plt.xlabel('State', fontsize=12)
            plt.ylabel('Number of Customers', fontsize=12)
            plt.xticks(rotation=45)
            plt.legend(title='Customer Status')

        elif column == 'Geography':
            sns.set(style="whitegrid", palette="muted")
            plt.figure(figsize=(16, 8))
            sns.countplot(data=data, x='State', hue='Customer_Status')
            plt.title('Customer Status Distribution by Geography (State)', fontsize=16)
            plt.xlabel('State', fontsize=12)
            plt.ylabel('Number of Customers', fontsize=12)
            plt.xticks(rotation=45)
            plt.legend(title='Customer Status')


        else:
            plt.figure(figsize=(10, 5))
            if column == "Gender":
                plt.figure(figsize=(4, 3))
            elif column == "Churn_Category":
                plt.figure(figsize=(8, 3))
                plt.xticks(rotation=70) 
            elif column == "State" or column == "Age":
                plt.figure(figsize=(16, 3))
                plt.xticks(rotation=70)

            sns.barplot(x=column, y='Count', hue='Customer_Status', data=gender_churn)
            plt.title(column + " Based Churned Graph")

        # Convert plot to base64
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', bbox_inches='tight')
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.close()

        context = {
            'data': column + " Based Churned Graph",
            'img': img_b64,
            'selected_filter': column
        }

        return render(request, 'Visualization.html', context)

    # If GET request, just show blank form
    return render(request, 'Visualization.html', {'selected_filter': None})



def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def about(request):
    if request.method == 'GET':
       return render(request, 'about.html', {})
