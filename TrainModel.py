import pandas as pd
from pycaret.regression import *
import numpy as np


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

    return df, dates_manquantes

def process_missing_values(data):
  data["PRCP"].interpolate(method='spline',
                           order=10,
                           inplace=True)
  return data


# Charger les données
df = pd.read_csv('data/Daily_Diif_30_days_Impute_Linear.csv')
df.drop(columns=["TAVG","TMAX" ,"TMIN","PRCP_0","TMAX_0","TMIN_0"], inplace=True)
df, empty_dates = clean_time_index(df)
df.set_index("DATE", inplace=True)


# Traiter les valeurs manquantes
df = process_missing_values(df)


s  = setup(
    data = df,
           target = 'PRCP',
           session_id=123,
    index = False,
    numeric_imputation="knn",
    numeric_iterative_imputer="lightgbm",
    categorical_iterative_imputer="knn",

           #date_features= ["Jour", "Mois", "Annee"]
    )


best = compare_models(
    include=['lr', 'lasso', 'ridge', 'en', 'knn', 'dt', 'rf'],
    fold = 5,
    sort = 'RMSE',
    n_select = 3,
)

tuned_best = [tune_model(i) for i in best]

blender = blend_models(estimator_list = tuned_best)

final_model = finalize_model(blender)

# Sauvégarder tous les models 
save_model(final_model, 'model')
