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
sns.set_theme()

def prepare_model():
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
    return dataset, rf_cls, label_encoder, sc, labels

def Predict(request):
    if request.method == 'GET':
        return render(request, 'Predict.html', {})


def ViewFile(request):
    if request.method == 'GET':
        return render(request, 'ChurnPrediction.html', {})

def PredictAction(request):
    if request.method == 'POST':
        dataset, rf_cls, label_encoder, sc, labels = prepare_model()
        try:
            myfile = request.FILES['t1'].read()
            fname = request.FILES['t1'].name
            file_path = "ChurnApp/static/" + fname

            if os.path.exists(file_path):
                os.remove(file_path)
            with open(file_path, "wb") as file:
                file.write(myfile)

            testdata = pd.read_csv(file_path)
            temp = testdata.values

            # Track columns with encoding errors
            invalid_columns = []

            for i in range(len(label_encoder)):
                le = label_encoder[i]
                column_name = le[0]
                if column_name != 'Customer_Status':
                    try:
                        testdata[column_name] = pd.Series(le[1].transform(testdata[column_name].astype(str)))
                    except ValueError:
                        invalid_columns.append(column_name)

            # If any invalid columns found, display all at once
            if invalid_columns:
                error_list = ', '.join(f"'{col}'" for col in invalid_columns)
                error_message = f"The uploaded file has incorrect or unseen values in the following column(s): {error_list}. Please correct them and try again."
                return render(request, 'Graph.html', {'error': error_message})

            testdata.fillna(dataset.mean(), inplace=True)
            testdata = testdata.values
            testdata = sc.transform(testdata)
            predict = rf_cls.predict(testdata)
            predict = predict.ravel()

            label_map = {
                "Churned": "Likely to churn",
                "Joined": "Just joined",
                "Stayed": "Likely to Stay"
            }

            table_data = []
            for i in range(len(predict)):
                original_label = labels[predict[i]]
                display_label = label_map.get(original_label, original_label)
                row_data = {
                    'test_data': temp[i],
                    'prediction': display_label,
                    'color_class': 'text-danger' if predict[i] == 0 else ('text-primary' if predict[i] == 1 else 'text-success')
                }
                table_data.append(row_data)

            return render(request, 'Graph.html', {'table_data': table_data})
        
        except Exception as e:
            return render(request, 'Graph.html', {'error': f"An error occurred: {str(e)}"})

    return render(request, 'Graph.html', {'error': "Invalid request method"})

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
