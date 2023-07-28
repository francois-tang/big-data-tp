import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import confusion_matrix


def center_reduce(data):
    # Calcul des moyennes et écart-types
    moyennes = data.mean(axis=0)
    ecart_types = data.std(axis=0, ddof=1)

    # Données centrées et réduites
    Xc = data - moyennes
    Xcr = Xc / ecart_types
    #Xcr.std(axis=0,ddof=0)
    Xcr = Xcr.add_suffix('_cr') # renommage des colonnes
    
    return Xcr


def perform_pca(data):
    # Réalisation de l'pca
    pca = PCA()
    CP = pca.fit(data)
    components = pca.fit_transform(data)
    
    # Mise en forme des composantes principales
    new_data = pd.DataFrame(data=components, columns=['CP_' + str(col) for col in range(1, pca.n_features_in_ + 1)])
    
    return pca, new_data


def plot_inertie(pca):
    """
    Affiche le graphique d'inertie expliquée en fonction du nombre de dimensions d'une ACP.
    
    Args:
        pca (PCA): L'objet PCA contenant les résultats de l'ACP.
    """
    plt.bar(np.arange(len(pca.explained_variance_ratio_))+1, pca.explained_variance_ratio_*100)
    plt.plot(np.arange(len(pca.explained_variance_ratio_))+1, np.cumsum(pca.explained_variance_ratio_*100), 'r--o')
    plt.xlabel("Dimensions", fontsize=14)
    plt.ylabel("% d'inertie expliquée", fontsize=14)
    plt.title("Inertie expliquée en fonction du nombre de dimensions", fontsize=14)
    plt.grid(True)
    plt.show()
    

def eval_individual_quality(Xpca, pca):
    qual = Xpca.values ** 2  # Contribution de chaque variable aux différentes composantes principales
    qual = (qual.T / qual.sum(axis=1)).T  # Ratio ligne
    df_qualite = pd.DataFrame(data=qual, columns=list(range(1, pca.n_features_in_+1)))
    df_qualite = df_qualite.add_prefix('CP_')  # Renommage des colonnes
    df_qualite = df_qualite * 100
    df_qualite_check = df_qualite.sum(axis=1)  # Vérification de la qualité de représentation des individus par les CP

    return df_qualite, df_qualite_check


def eval_contribution(Xpca, pca):
    contr = Xpca.values ** 2
    contr = contr / np.sum(contr, axis=0) # Ratio colonne
    df_contribution = pd.DataFrame(data=contr, columns=list(range(1, pca.n_features_+1)))
    del contr
    df_contribution = df_contribution.add_prefix('CP_') # Renommage des colonnes
    df_contribution = df_contribution * 100
    df_contribution_check = df_contribution.sum(axis=0) # vérification de la contribution des individus aux CP

    return df_contribution, df_contribution_check


def eval_correlations(data, Xpca):
    # Calcul des corrélations entre les anciennes et les nouvelles variables
    corrOldNew = np.corrcoef(data.T, Xpca.T)
    corrOldNew = corrOldNew[0:len(data.columns), len(data.columns):]
    
    # Création du DataFrame des corrélations
    coordonneesDesVariables = pd.DataFrame(data=corrOldNew, index=variables, columns=list(range(1, Xpca.shape[1] + 1)))
    coordonneesDesVariables = coordonneesDesVariables.add_prefix('CP_')  # Renommage des colonnes
    
    return coordonneesDesVariables


def plot_all_correlation_circle(coordonneesDesVariables, variables):
    d = coordonneesDesVariables.shape[1]  # Nombre de dimensions

    # Coordonnées maximales de chacune des figures
    x_lim = [-1.1, 1.1]
    y_lim = [-1.1, 1.1]
    cpt = 0

    fig, axs = plt.subplots(figsize=(10, 10 * d))

    for i in range(d - 1):
        for j in range(i + 1, d):
            cpt += 1
            ax = plt.subplot(int(d * (d - 1) / 2), 1, cpt)

            # Cercle unitaire
            cercle = plt.Circle((0, 0), 1, color='red', fill=False)
            ax.add_artist(cercle)

            # Projection du nuage des variables
            for k in range(len(variables)):
                ax.arrow(0, 0, coordonneesDesVariables.iloc[k, i], coordonneesDesVariables.iloc[k, j],
                         length_includes_head=True, head_width=0.05, head_length=0.1, fc='k', ec='k')
                # Ornementation
                plt.text(coordonneesDesVariables.iloc[k, i], coordonneesDesVariables.iloc[k, j], variables[k])

            plt.title('Axes {} et {}'.format(i + 1, j + 1))

            # Ajout d'une grille
            plt.grid(color='lightgray', linestyle='--')

            # Ajouter des deux axes correspondants aux axes factoriels
            ax.arrow(x_lim[0], 0, x_lim[1] - x_lim[0], 0, length_includes_head=True, head_width=0.05, head_length=0.1,
                     fc='k', ec='k')
            plt.plot(plt.xlim(), np.zeros(2), 'k-')
            plt.text(x_lim[1], 0, "axe {:d}".format(i + 1))

            ax.arrow(0, y_lim[0], 0, y_lim[1] - y_lim[0], length_includes_head=True, head_width=0.05, head_length=0.1,
                     fc='k', ec='k')
            plt.plot(np.zeros(2), plt.ylim(), 'k-')
            plt.text(0, y_lim[1], "axe {:d}".format(j + 1))

            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()
    

def plot_correlation_circle(cp1, cp2, coordonneesDesVariables):
    plt.figure(figsize=(10, 10))
    cercle = plt.Circle((0, 0), 1, color='red', fill=False)
    plt.gca().add_artist(cercle)
    
    for i, var in enumerate(variables):
        plt.arrow(0, 0, coordonneesDesVariables.loc[var, cp1], coordonneesDesVariables.loc[var, cp2],
                  color='k', alpha=0.8, head_width=0.05, head_length=0.1, length_includes_head=True, fc='k', ec='k')
        plt.text(coordonneesDesVariables.loc[var, cp1], coordonneesDesVariables.loc[var, cp2], var,
                 ha='center', va='center')
        plt.text(1.1, 0, f"Axe : {cp1}")
        plt.text(0,1.1, f"Axe : {cp2}")
    
    plt.title(f'Cercle de Corrélation des composantes : {cp1} et {cp2}')
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.grid(color='lightgray', linestyle='--')
    
    # Ajout des axes
    plt.arrow(-1.1, 0, 2.2, 0, length_includes_head=True, head_width=0.05, head_length=0.1, color='k')
    plt.arrow(0, -1.1, 0, 2.2, length_includes_head=True, head_width=0.05, head_length=0.1, color='k')
    
    plt.show()


def elbow_method(data, max_clusters):
    # Liste des valeurs de K à tester
    k_values = range(1, max_clusters + 1)
    inertias = []

    # Calcul de l'inertie pour chaque valeur de K
    for k in k_values:
        model = KMeans(n_clusters=k)
        model.fit(data)
        inertias.append(model.inertia_)

    # Tracé de la courbe d'inertie
    plt.plot(k_values, inertias, 'bx-')
    plt.xlabel('Nombre de clusters (K)')
    plt.ylabel('Inertie')
    plt.title('Méthode du coude pour choisir le nombre de clusters')
    plt.show()


def silhouette_method(data, max_clusters):
    # Liste des valeurs de K à tester
    k_values = range(2, max_clusters + 1)
    silhouette_scores = []

    # Calcul du score de silhouette pour chaque valeur de K
    for k in k_values:
        kmeans = KMeans(n_clusters=k)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    # Tracé de la courbe des scores de silhouette
    plt.plot(k_values, silhouette_scores, 'bx-')
    plt.xlabel('Nombre de clusters (K)')
    plt.ylabel('Score de silhouette moyen')
    plt.title('Méthode de la silhouette pour choisir le nombre de clusters')
    plt.show()


def compute_metrics(confusion_matrix):
    # Calcul de la précision
    precision = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[0, 1])
    
    # Calcul du rappel (recall)
    recall = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[1, 0])
    
    # Calcul du score F1
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    # Calcul du taux de classification correcte (accuracy)
    accuracy = (confusion_matrix[0, 0] + confusion_matrix[1, 1]) / np.sum(confusion_matrix)
    
    return precision, recall, f1_score, accuracy


# Charger les données
data = pd.read_csv(r'C:\Users\ludwi\Desktop\BIGDATA-MIAGE-G2\diabetes.csv')

# Nom des variables
variables = list(data.columns)

#data = data.drop('Outcome', axis=1)
k = 80  # Nombre de clusters souhaité

# Méthode du coude
elbow_method(data, k)

# Méthode silhoutte
silhouette_method(data, k)

k = 150  # Nombre de clusters souhaité

# Application de K-means avec k clusters
kmeans = KMeans(n_clusters=k)
kmeans.fit(data)
kmeans_labels = kmeans.labels_

# Récupération du dataset réduit
data_with_cluster = data.copy()
data_with_cluster['Cluster'] = kmeans_labels

# Automatisation de la création des noms de clusters
cluster_names = {}
for i in range(k):
    cluster_names[i] = f"Cluster {i+1}"  # Cluster i+1

# Utiliser la méthode "map" pour attribuer les nouveaux noms aux clusters
data_with_cluster['Cluster'] = data_with_cluster['Cluster'].map(cluster_names)

# Récupérez l'inertie de chaque cluster
distances = kmeans.transform(data) # distances entre chaque échantillon et chaque centroïde
inertias = np.sum(distances**2, axis=0) # inertie de chaque cluster en utilisant les distances

# Récupérez l'index du cluster ayant le plus d'inertie
cluster_max_inertia = np.argmax(inertias)

    
# Évaluation de K-means
print(f"Indice de silhouette : {silhouette_score(data, kmeans_labels):.4f}")
print(f"Inertie totale : {kmeans.inertia_:.2f}")

'''
'''
data = pd.DataFrame(kmeans.cluster_centers_,columns=data.columns)

# Données centrées, réduites
Xcr = center_reduce(data)

# Matrice de corrélations
ax = sns.heatmap(Xcr.corr(), annot=True, fmt=".2f", linewidths=.5, vmin=-1, vmax=1)

# diagramme de dispersion
g = sns.pairplot(Xcr, diag_kind="kde", markers="+",
                  plot_kws=dict(s=50, facecolor="b", linewidth=1),
                  diag_kws=dict(fill=True))

# Réalisation de l'pca
pca, Xpca = perform_pca(Xcr)

# Inertie expliquée en fonction du nombre de composantes principales
plot_inertie(pca)

# Qualité de représentation des individus [profil ligne]
df_qualite, df_qualite_check = eval_individual_quality(Xpca, pca)

# Contribution des individus (aux nouvelles composantes) en % [profil colonne]
# (ratio entre l'inertie de l'individu sur l'axe et l'inertie totale de l'axe)
df_contribution, df_contribution_check = eval_contribution(Xpca, pca)

# Corrélations en anciennes et nouvelles variables
Xcorr = eval_correlations(data, Xpca)

# nombres de dimensions
pourcentageDInertieSeuil = 95
d = np.argmax(np.cumsum(pca.explained_variance_ratio_)>=pourcentageDInertieSeuil/100)+1
print("Nombres de dimensions (>={:.0f}% inertie) : ".format(pourcentageDInertieSeuil),d)

# Construction des cercles de corrélations
plot_all_correlation_circle(Xcorr, variables)

# Créer le graphique du cercle de corrélation    
plot_correlation_circle('CP_1', 'CP_2', Xcorr)  
'''
# Méthode du coude
elbow_method(Xpca, 13)

# Méthode silhoutte
silhouette_method(Xpca, 13)

# Application de K-means
kmeans = KMeans(n_clusters=2)  # Spécifier le nombre de clusters souhaité
kmeans.fit(Xpca)
kmeans_labels = kmeans.labels_

# Visualisation des clusters obtenus
plt.scatter(Xpca.CP_1, Xpca.CP_2, c=kmeans_labels)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Clustering des données diabetes avec K-means après l\'pca')
plt.show()

# Récupération du dataset réduit
data_with_cluster = data.copy()
data_with_cluster['Cluster'] = kmeans_labels

# Renommer les clusters en utilisant un dictionnaire de correspondance
cluster_names = {
    0: 1,#"Cluster A",
    1: 0,#"Cluster B"
}

# Utiliser la méthode "map" pour attribuer les nouveaux noms aux clusters
data_with_cluster['Cluster'] = data_with_cluster['Cluster'].map(cluster_names)

# visualiser les cluster avec les composantes
Xpca_with_cluster = pd.concat([Xpca, pd.Series(kmeans_labels, name='Cluster')], axis=1)
# Tracer le diagramme de dispersion en colorisant les points selon les clusters
g = sns.pairplot(Xpca_with_cluster, hue='Cluster', diag_kind='kde', markers='+',
                 plot_kws=dict(s=50, linewidth=1), diag_kws=dict(fill=True))
'''
## Matrice de confusion et métriques
precision, recall, f1_score, accuracy = compute_metrics(confusion_matrix(data_with_cluster.Outcome, data_with_cluster.Cluster))
print(f"precision : {precision}")
print(f"recall : {recall}")
print(f"f1_score : {f1_score}")
print(f"accuracy : {accuracy}")
