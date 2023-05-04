# Importation des bibliotheques de Python -----------------------------------------------------------------------------------------------------
from sklearn.model_selection import LeaveOneOut
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from itertools import combinations
from numpy import mean
from sklearn.model_selection import cross_val_score



# Traitement des donnees -----------------------------------------------------------------------------------------------------------------------

# Lecture des donnees

# root est le chemin d'acces de la BDD depuis l'ordinateur 
root = 'nom_du_dataset'

# Transformation de la BDD en dataframe 

# df contient toutes les colonnes et toutes les lignes de la BDD 
df = pd.read_excel(root)
# df1 contient toutes les lignes et les colonnes des 6 biomarqueurs, de temps_de_survie et de vivant_ou_non de la BDD  
df1 = pd.concat([df['Biomarqueur_1'], df['Biomarqueur_2'], df['Biomarqueur_3'], df['Biomarqueur_4'], df['Biomarqueur_5'], df['Biomarqueur_6'], df['temps_de_survie'], df['vivant_ou_non']], axis=1)


# Biom est la liste des diffÃ©rents biomarqueurs de la BDD
Biom = ['Biomarqueur_1','Biomarqueur_2','Biomarqueur_3','Biomarqueur_4','Biomarqueur_5', 'Biomarqueur_6']

# Charger les donnÃ©es de la BDD

# condition est la condition appliquÃ©e sur le temps_de_survie et vivant_ou_non afin de prendre en compte les N patients possédant soit un temps_de_survie > D avec un vivant_ou_non = 1 ou à  0 soit un temps_de_survie < D avec un vivant_ou_non = 1
condition = ((df['temps_de_survie'] < D) & (df['vivant_ou_non'] == 1)) | ((df['temps_de_survie'] > D) & (df['vivant_ou_non'] == 1)) | ((df['temps_de_survie'] > D) & (df['vivant_ou_non'] == 0))
# On applique la condition sur le dataframe df1 
L = df1[condition]


# Création des dataframes X (variables de caractéristiques ie 6 biomarqueurs) et Y (cible ie le temps_de_survie)
X = pd.concat([L['Biomarqueur_1'], L['Biomarqueur_2'], L['Biomarqueur_3'], L['Biomarqueur_4'], L['Biomarqueur_5'], L['Biomarqueur_6']], axis=1)
Y = pd.concat([L['temps_de_survie']])

# On retient le nom des colonnes 
column_names = X.columns.tolist()


# Appliquer la transformation de normalisation sur les données X
#X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=column_names)

# La fonction label permet de distinguer les classes de temps_de_survie des patients
# Defav si temps_de_survie < D et Fav si temps_de_survie > D 
def label(Y):
    if Y<D:
        return 'Defav'
    else:
        return 'Fav'


# On met à jour notre dataframe cible Y. Il est maintenant sous forme catégorielle
Y = pd.Series([label(k) for k in Y ])
 
#---------------------------------------------------------------------------------------------------------------------

# La fonction resultats() permet d'afficher la grille des inputs des couples de biomarqueurs et d'avoir la liste scores des biomarqueurs
# resultats() est la liste des scores 
def resultats():   
    # card_biom est le nombre de biomarqueurs que l'on veut comparer
    card_biom = 2
    # S est la liste des scores de performance de chaque comparaison entre les biomarqueurs
    S = []
    
    # Obtenir tous les sous-ensembles de nombre card_biom elements dans l'ensemble Biom
    sous_ensembles = set(combinations(Biom, card_biom))
    sous_ensembles = list(sous_ensembles) #convertir en liste le sous-ensemble 
    
       
    # Parcourir tous les sous-ensembles
    for i, k in enumerate(sous_ensembles):
        
        # Extraction des colonnes du DataFrame initial
        X_bis = X.loc[:, [(k[0]), (k[1])]]
        
            

        # Creation du modèle l'arbre decision
        # max_depth : profondeur de l'abre voulu 
        model = DecisionTreeClassifier(criterion='gini', max_depth=4) 
   
    
        #Evaluer le modele par validation croisée
        cv = LeaveOneOut() # methode du leave one out 
        scores = cross_val_score(model, X_bis, Y,scoring='balanced_accuracy' ,cv=cv, n_jobs=-1)
    
    
        # Evaluation de la performance de l'arbre de decision (mise en forme des scores avec les couples de biomarqueurs)
        a = "(" + k[0]+", "+k[1]+")" +": "+ str(round(mean(scores),3))
        S.append(a)
    
 
    
    # Trier la liste des scores 
    # result contient les couples de biomarqueurs avec le score de performance rangé dans l'odre décroissant 
    result = sorted(S, key=lambda x: float(x.split(': ')[-1]), reverse=True)
    
    
    # Traitement des résultats pour pouvoir afficher les arbres de décisions
    # Initialiser une liste vide pour stocker les chaines de caracteres extraites
    mesures = []
    
    # Boucler sur chaque element de la liste result
    for element in result:
        # Extraire les noms des mesures Ã  partir de l'Ã©lÃ©ment actuel
        mesures_actuelles = element.replace('(', '').replace(')', '').split(':')[0].split(',')
        # Retirer les espaces en dÃ©but et fin de chaque nom de mesure
        mesures_propres = [mesure.strip() for mesure in mesures_actuelles]
        # Extraire le nombre aprÃ¨s les deux points
        nombre = float(element.split(':')[-1].strip())
        # Ajouter les noms de mesure et le nombre extrait Ã  la liste mesures
        mesures.append(tuple(mesures_propres + [nombre]))
    return mesures



# La fonction plot_graph_decision_tree(X, y) permet d'afficher l'arbre de décision d'un couple donnée 
# var1 et var2 sont les deux biomarqueurs sous forme de chaines de caracteres dont on veut l'abre de decision
def plot_decision_tree(var1, var2, mesure_list = resultats()):
    
    # Trouver l'entree dans mesure_list correspondant aux noms de variable donnees
    mesure = None
    for m in mesure_list:
        if m[0] == var1 and m[1] == var2:
            mesure = m
            break
    if mesure is None:
        print(f"Aucune mesure trouvee pour les variables {var1} et {var2}")
        return
    
    # extraire les donnees correspondantes
    X_train = X.loc[:, [var1, var2]]
    y_train = Y
    
    # Modèle mathématique 
    # max_depth : profondeur de l'abre voulu 
    model = DecisionTreeClassifier(criterion='gini', max_depth=4)
    # Entrainer le modele
    model.fit(X_train, y_train)
    
    
    # tracer l'arbre de decision
    # figsize permet de modifier la largeur et la hauteur de l'image de l'arbre de décision 
    fig, ax = plt.subplots(figsize=(40, 40))
    tree.plot_tree(model, filled=True, ax=ax, fontsize=19, feature_names=X_train.columns)
    
    # ajouter le titre a  l'arbre
    ax.set_title(f"Arbre de decision pour le couple: ({var1}, {var2})\n Balanced Accuracy: {mesure[2]}", fontsize=36)
    
    # afficher le graphe
    plt.show()
    

# La fonction predictions permet de prédire la classe d'un nouveau patient à partir des biomarqueurs var1 et var2 (sous forme de chaines de caractere)
# profondeur_arbre est fixé à 4 par défaut
def predictions(var1, var2, X_test, profondeur_arbre=4):
    
    # Extraction des colonnes des deux biomarqueurs dans le dataframe de test 
    X_pred = X_test.loc[:, [var1, var2]]
    
    # Modèle mathématique
    # extraire les donnees correspondantes aux deux biomarqueurs pour l'entrainement du modele 
    X_train = X.loc[:, [var1, var2]]
    y_train = Y
    
    # Entrainer le modele de DecisionTreeClassifier
    model = DecisionTreeClassifier(criterion='gini', max_depth=profondeur_arbre)
    model.fit(X_train, y_train)
    

    # Faire la prédiction de la classe du dataframe de test 
    y_pred = model.predict(X_pred)
    
    return y_pred
    
    
