import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

URL = "https://data.seattle.gov/api/views/65db-xm6k/rows.csv?accessType=DOWNLOAD"
local_file = "data/fremont.csv"

data = pd.read_csv(local_file) # on transforme le data en dataframe
print(data.shape) # affiche les dimension du tableau
print(data.head) # affiche le début du tableau 


############# doublons #########################################


data.drop_duplicates('Date',keep='last', inplace=True) # on supprime les doublons en lui disant que si les dates sont les même (dexu vélo qui passent exactement en même temps c'est forcément un doublon)
# on garde la derniere des ligne en doublon et on met inplace=True pour pas que il crée de copie
print(data.shape)


############# échauffement le DatetimeIndex ###################


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


############# parser les dates ###############################


print(data.info())
print(data.dtypes) 

data.index = pd.to_datetime(data.Date, format= '%m/%d/%Y %I:%M:%S %p') # on crée une colonne DatetimeIndex de date
del data['Date'] # on supprime la colonne de date qui n'est pas un datetimeindex


############# renommons les colonnes #########################


data.columns = ['Total','West','East']


############# données manquantes et extension types ##########


#1
print(data[data.isna().any(axis=1)].shape) 

#2
data = data.convert_dtypes(convert_integer=True)


############# à quoi ça ressemble ###########################


sns.set_theme(rc={'figure.figsize': (12, 4)})
data[['East', 'West']].plot()



############ changer la fréquence temporelle ################


#1
data.resample("1W").sum().plot(figsize = (12,4), title="fréquence par semaine")
plt.show()

#2
if data.shape[0] <= data.resample("1W").sum().shape[0]*7*24:
    print(True)
else:
    print(False)

############ découpage temporel #############################


data = data[data.index <= "2017-01-01 00:00:00"]


############ évolution annuelle #############################


data.resample("1d").sum().rolling(365).sum().plot( # on indique que on fait la somme sur les jours, pour chaque année
    figsize = (12,4), 
    title="évolution anuelle", 
    ylim=0) # on indique que l'axe des y doit commencer a 0
plt.show()


########### profil journalier moyen #########################


data.groupby(data.index.time).sum().plot( # on indique que on veut s'interesser aux heures de passage (de maniere indépendante de la date)
    figsize = (12,4),                     # on fait la somme des valeurs pour avoir le profil sur une suele journée
    title="profil journalier moyen" 
)
plt.show()


########### une courbe par jour : la pivot_table ##########


pivoted = data.pivot_table('Total', index= data.index.time, columns=data.index.date) # on crée notre tableau qui prend en ligne les heures de la journée et en colonne les differentes journées
pivoted.plot (
    legend = None, # on lui dit de ne pas afficher la légende qui est énooorme
    alpha = 0.01, # on baisse l'opacité des courbes pour faire apparaître les tendances par superpositions
    figsize = (12,4),  # on va obtenir un graphique de plein de courbe ressemblant au graphique précédent (lui faisait la somme de toutes les courbes affichées là)
    title ="une courbe par jour"
)
plt.show()


########### classification des jours ######################


pca_input = pivoted.T.fillna(0) # on fait la transposé de la matrice d'avant et on rempli par des zeros les cellules sans information
pca_input.plot( 
    legend = None, 
    alpha = 0.5,
    figsize = (12,4),
    title = "classification des jours"
)
plt.show()


########### ACP ###########################################


from sklearn.decomposition import PCA

pca = PCA(n_components = 2).fit_transform(pca_input) # on utilise la boite noir
plt.scatter(pca[:, 0], pca[:, 1], alpha=0.7)
plt.show() # on plot et on voit que cela differentie les deux types de jours selon deux droite


########### Gaussian mixture ##############################


from sklearn.mixture import GaussianMixture

mg = GaussianMixture(2)
labels = mg.fit(pca_input).predict(pca_input) # on trie les jours en deux groupe et on leur attribue des labels 
plt.scatter(pca[:,0],pca[:,1], c=labels, cmap='rainbow') # on plot les deux colonnes et les couleurs sont faites en fonction des labels que l'on avait deja attribué aux jours
plt.colorbar()
plt.show()

## comparons chaque famille aux profils journaliers

pivoted.loc[:,labels==0].plot(legend=False, title = "famille label = 0", alpha=0.01) # on affiche que la colonne qui nous interesse
plt.show()


pivoted.loc[:,labels==1].plot(legend=False, title = "famille label = 1", alpha=0.01)
plt.show()

    # on observe que la famille 0 represente une bosse avec un maximum vers 12h00 --> correspond probablement aux weekends
    # la famille 1 représente les deux pics --> les gens qui vont travailler le matin et qui revienne le soir pendant la semaine

## vérifications avec le jour de la semaine


jour_semaine = pd.DatetimeIndex(pivoted.columns).day_of_week # on transforme en DateTimeIndex et on met un indice en fonction du jour de la semaine
plt.scatter(pca[:,0], pca[:,1], c=jour_semaine, cmap = 'rainbow')
plt.colorbar()
plt.show()

## les moutons noirs

dates_bon_format = pd.DatetimeIndex(pivoted.columns)
odd_index = (labels ==1) & (jour_semaine <5) # on selectionne les points qui appartiennent a la famille 1 mais qui sont pendant la semaine
odd_dates = dates_bon_format[odd_index] # on fait la liste des dates dont les indices corresponde aux points qui jurent
pivoted[odd_dates].plot(legend=False, alpha=0.5) # on plot les jours qui font partie de la liste que l'on vient de crée
plt.show()








