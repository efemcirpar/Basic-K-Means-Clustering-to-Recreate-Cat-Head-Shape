import pandas as pd
import matplotlib.pyplot as plt

cats = pd.read_csv('cat.txt', delimiter=' ')
cats['Cluster'] = cats['Cluster'].str.strip('\t')

plt.scatter(cats['Dimension1'][(cats['Cluster'] == 'Head')], cats['Dimension2'][(cats['Cluster'] == 'Head')], marker='o', color='blue',  label='Head')
plt.scatter(cats['Dimension1'][(cats['Cluster'] == 'Ear_left')], cats['Dimension2'][(cats['Cluster'] == 'Ear_left')], marker='o', color='orange', label='Ear_left')
plt.scatter(cats['Dimension1'][(cats['Cluster'] == 'Ear_right')], cats['Dimension2'][(cats['Cluster'] == 'Ear_right')], marker='o', color='black', label='Ear_right')
plt.title('First Clusters')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.legend()
plt.show()

from sklearn.cluster import KMeans

cat_part = cats['Cluster']
X = cats.drop(['Cluster'],axis=1) 
kmeans = KMeans(n_clusters=3, random_state=0)
catpart_KMeans = kmeans.fit_predict(X)
plt.scatter(cats['Dimension1'][(catpart_KMeans == 0)], cats['Dimension2'][(catpart_KMeans == 0)], marker='o', color='blue', label='Cluster 0')
plt.scatter(cats['Dimension1'][catpart_KMeans == 1], cats['Dimension2'][catpart_KMeans == 1], marker='o', color='black', label='Cluster 1')
plt.scatter(cats['Dimension1'][catpart_KMeans == 2], cats['Dimension2'][catpart_KMeans == 2], marker='o', color='orange', label='Cluster 2')
plt.title('K-Means')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.legend()
plt.show()
input()
