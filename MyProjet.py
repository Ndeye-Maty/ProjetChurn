import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import io 

from sklearn.preprocessing import LabelEncoder



def load_data():
    data = pd.read_csv('Expresso_churn_dataset.csv')
    return data

data = load_data()
# Afficher les informations générales
st.title('Exploration du jeu de données Expresso Churn')


buffer = io.StringIO()
data.info(buf=buffer)
info_string = buffer.getvalue()
st.text(info_string)

# 
# Gérer les valeurs manquantes et corrompues
data.dropna(inplace=True)
st.write("Nombre de lignes après suppression des valeurs manquantes : ", data.shape[0])

st.write(data.isnull().sum())

# Supprimer les doublons
data.drop_duplicates(inplace=True)


def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

for column in data.select_dtypes(include=['float64', 'int64']).columns:
    data = remove_outliers(data, column)

st.write("Nombre de lignes après suppression des valeurs aberrantes : ", data.shape[0])

# Encoder les caractéristiques catégorielles
st.header('Encodage des caractéristiques catégorielles')
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    if data[column].isnull().sum() == 0:  # Encoder seulement si aucune valeur manquante
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

st.write("Aperçu des données après encodage :")
buffer = io.StringIO()
data.info(buf=buffer)
info_string = buffer.getvalue()
st.text(info_string)


# Supprimer les colonnes non nécessaires
columns_to_drop = ['user_id', 'ZONE1', 'ZONE2', 'TOP_PACK']  # Ajustez cette liste en fonction de votre exploration
X = data.drop(columns_to_drop + ['CHURN'], axis=1)
y = data['CHURN']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Former un classificateur pour identifier les caractéristiques importantes
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Identifier les caractéristiques les plus importantes
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
important_features = feature_importances.head(10).index.tolist()  # Sélectionner les 10 caractéristiques les plus importantes

# Réduire le jeu de données aux caractéristiques importantes
X = X[important_features]

# Diviser à nouveau les données en ensembles d'entraînement et de test avec les caractéristiques importantes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Former le modèle final avec les caractéristiques importantes
model.fit(X_train, y_train)

# Évaluer le modèle
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Afficher l'exactitude du modèle
st.write(f'Exactitude du modèle: {accuracy:.2f}')

# Ajouter des champs de saisie pour les caractéristiques importantes
st.header('Faire une prédiction')
input_data = {}
for column in important_features:
    value = st.number_input(f'Entrer la valeur pour {column}', value=0)
    input_data[column] = value

# Bouton de validation pour faire des prédictions
if st.button('Faire une prédiction'):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.write(f'Prédiction: {"Churn" if prediction[0] == 1 else "No Churn"}')
    st.write(f'Probabilité de Churn: {prediction_proba[0][1]:.2f}')

#Fin du projet 

 
