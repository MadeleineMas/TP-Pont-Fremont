import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

URL = "https://data.seattle.gov/api/views/65db-xm6k/rows.csv?accessType=DOWNLOAD"
local_file = "data/fremont.csv"

data = pd.read_csv(local_file) # on transforme le data en dataframe
print(data.shape) # affiche les dimension du tableau
print(data.head) # affiche le début du tableau 

############# doublons 

data.drop_duplicates('Date',keep='last', inplace=True) # on supprime les doublons en lui disant que si les dates sont les même (dexu vélo qui passent exactement en même temps c'est forcément un doublon)
# on garde la derniere des ligne en doublon et on met inplace=True pour pas que il crée de copie
print(data.shape)

############# échauffement le DatetimeIndex

sample_dates = pd.to_datetime([
    "2024-01-15 08:00:00",
    "2024-01-20 17:30:00",
    "2024-02-03 12:00:00",
])

#1 sample_dates est un DatetimeIndex, serie de date et d'heure

#2 
    # sample_dates.date renvoie la liste des différentes dates du tableau
    # sample_date.time renvoie la liste des différentes heures du tableau
    # sample_dates.dayofweek renvoie Index([0, 5, 5], dtype='int32'), il donne donc les jours de la semaine associé a chaque date

#3 
sample_dates[sample_dates >= "2024-01-20 00:00:00"]

#4 
sample_dates[(sample_dates >= "2024-01-1 ") & (sample_dates <= "2024-01-31 ") ]

############# parser les dates

print(data.info())
print(data.dtypes) 

data.index = pd.to_datetime(data.Date, format= '%m/%d/%Y %I:%M:%S %p') # on crée une colonne DatetimeIndex de date
del data['Date'] # on supprime la colonne de date qui n'est pas un datetimeindex

############# renommons les colonnes

data.columns = ['Total','West','East']

############# données manquantes et extension types

#1
print(data[data.isna().any(axis=1)].shape) 

#2
data = data.convert_dtypes(convert_integer=True)

############# à quoi ça ressemble

sns.set_theme(rc={'figure.figsize': (12, 4)})
data[['East', 'West']].plot()


############ changer la fréquence temporelle 

#1
data.resample("1W").sum().plot(figsize = (12,4), title="fréquence par semaine")
plt.show()

#2
if data.shape[0] <= data.resample("1W").sum().shape[0]*7*24:
    print(True)
else:
    print(False)

############ découpage temporel

data = data[data.index <= "2017-01-01 00:00:00"]

############ évolution annuelle

data.resample("1d").sum().rolling(365).sum()