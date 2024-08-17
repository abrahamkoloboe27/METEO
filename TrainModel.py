import pandas as pd
from pycaret.regression import *
import numpy as np
import os
import shutil

# Créer un répertoire pour les modèles s'il n'existe pas
os.makedirs('models', exist_ok=True)



def clean_time_index(data):
    df = data.copy()
  # Assurez-vous que la colonne "Date" est de type datetime
    df["DATE"] = pd.to_datetime(df["DATE"])

  # Créez un index avec la colonne "Date" pour permettre la manipulation de dates
    df.set_index("DATE", inplace=True)

    # Créez un DataFrame avec toutes les dates quotidiennes dans la plage
    min_date = df.index.min()
    max_date = df.index.max()
    jours_complets = pd.date_range(start=min_date, end=max_date, freq='D')

    dates_manquantes = set(jours_complets) - set(df.index)

    # Ajoutez les dates manquantes au DataFrame avec des valeurs NaN
    for Date in dates_manquantes:
      df.loc[Date] = np.nan

    # Triez le DataFrame par l'index (Date) pour le remettre en ordre
    df.sort_index(inplace=True)

    # Réinitialisez l'index pour le ramener en colonne
    df.reset_index(inplace=True)

    return df

def process_missing_values(data):
    data["PRCP"].interpolate(method='spline',
                           order=3,
                           inplace=True,
                           )
    return data


# Charger les données
df = pd.read_csv('data/Daily_Diif_30_days_Impute_Linear.csv')
df
print("Fichier chargé")

df.drop(columns=["TAVG","TMAX" ,"TMIN","PRCP_0","TMAX_0","TMIN_0"], inplace=True)
print("Colonnes inutiles supprimées")

# Traiter les dates manquantes
df = clean_time_index(df)
print("Dates manquantes traitées")

# Mettre à jour l'index
df.set_index("DATE", inplace=True)
print("Index mis à jour")


# Traiter les valeurs manquantes
df = process_missing_values(df)
print("Valeurs manquantes traitées")

df.columns 
# Affficher le pourcentage de valeurs manquantes par colonne
missing_values = df.isna().mean() * 100
missing_values
print("Pourcentage de valeurs manquantes par colonne affiché")

# Supprimer les lignes de la colone 'PRCP' qui ont des valeurs manquantes
df.dropna(subset=['PRCP'], inplace=True)

# Sauvegarder le fichier
df.to_csv('data/Daily_Diif_30_days_Impute_Linear_Clean.csv', index=False)


# Initialiser l'environnement PyCaret
s  = setup(
            data = df,
           target = 'PRCP',
           session_id=123,
            index = False,
            fold = 10,
            numeric_imputation="knn",
            numeric_iterative_imputer="lightgbm",
            categorical_iterative_imputer="knn",
            normalize=True,
            transformation=True,
            experiment_name='SAVE_TrainModel_1',
            log_experiment = "mlflow",
   
    )
print("Setup terminé")


best = compare_models(fold = 3,
    sort = 'RMSE',
    n_select = 5,
    turbo = False
)
print("Modèles comparés")

# tuned_best = [tune_model(i) for i in best]
print("Modèles tunés")

blender = blend_models(estimator_list = tuned_best)
print("Modèles mélangés")

final_model = finalize_model(blender)
print("Modèle finalisé")

final_model_best = finalize_model(best[0])
# Sauvégarder tous les models 
save_model(final_model, 'model')
print("Modèle sauvegardé")
save_model(final_best , "models/final-best")

# Sauvegarder tous les models
for i in tuned_best:
  save_model(i, f'models/model_{i}')
  print (f"Modele.{i} sauvegardé ")

# Sauvegarder l'expérience
save_experiment('SAVE_TrainModel_1')


# Zipper le dossier models
shutil.make_archive('models', 'zip', 'models')


