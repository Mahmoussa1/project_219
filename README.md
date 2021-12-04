# project_219
#linear regression and clustering
import pandas as pd
phik=pd.read_csv('poro_perm_data.csv')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
#cleaning data set before linear regression
phik.isna().sum()
phik_new=phik.dropna()
phik_new.isna().sum()
#boolean indexing for porosity
logip = phik['Porosity (%)'] < 0
# Reassign as nan
phik.loc[logip,'Porosity (%)'] = np.nan

logip = phik['Porosity (%)'] > 100
# Reassign as nan
phik.loc[logip,'Porosity (%)'] = np.nan
phik['Porosity (%)']=phik['Porosity (%)']/100

#dropping nans
phik_new=phik.dropna()
#testing if there are still nans
phik_new.isna().sum()

#linear regression 1
por=np.array([phik_new['Porosity (%)']]).T
perm=np.array([phik_new['Permeability (mD)']]).T

model = LinearRegression()
model.fit(por, perm)
#calculating coefficient of determination
r_sq_poro_perm = model.score(por, perm)
perm_pred = model.predict(por)
plt.scatter(por, perm)
plt.plot(por,perm_pred, color="k")
plt.xlabel('Porosity')
plt.ylabel('Permeability')
plt.show()

#overbanks
overbanks=phik_new[phik_new.loc[:,'Facies']=="'overbanks'"]
ox=overbanks.loc[:,'Porosity (%)'].to_numpy().reshape(-1,1)
oy=overbanks.loc[:,'Permeability (mD)'].to_numpy().reshape(-1,1)
plt.scatter(ox,oy)
plt.xlabel('Porosity (%)')
plt.ylabel('Permeability (mD)')
plt.title('Porosity vs. Permeability, Overbanks')
plt.show()
model = LinearRegression()
model.fit(ox, oy)
r_sq = model.score(ox, oy)
perm_pred1 = model.predict(ox)
plt.plot(ox,perm_pred1, color="k")
print('The coefficient of determination is {}' .format(r_sq))

#crevasse splay
cs=phik_new[phik_new.loc[:,'Facies']=="'crevasse splay'"]
ox1=cs.loc[:,'Porosity (%)'].to_numpy().reshape(-1,1)
oy1=cs.loc[:,'Permeability (mD)'].to_numpy().reshape(-1,1)
plt.scatter(ox1,oy1)
plt.xlabel('Porosity (%)')
plt.ylabel('Permeability (mD)')
plt.title('Porosity vs. Permeability, Crevasse splay')
plt.show()
model = LinearRegression()
model.fit(ox1, oy1)
r_sq1 = model.score(ox1, oy1)
perm_pred2 = model.predict(ox1)
plt.plot(ox1,perm_pred2, color="k")
print('The coefficient of determination is {}' .format(r_sq1))

#channels
ch=phik_new[phik_new.loc[:,'Facies']=="'channel'"]
ox2=ch.loc[:,'Porosity (%)'].to_numpy().reshape(-1,1)
oy2=ch.loc[:,'Permeability (mD)'].to_numpy().reshape(-1,1)
plt.scatter(ox2,oy2)
plt.xlabel('Porosity (%)')
plt.ylabel('Permeability (mD)')
plt.title('Porosity vs. Permeability, Channel')
plt.show()
model = LinearRegression()
model.fit(ox2, oy2)
r_sq2 = model.score(ox2, oy2)
perm_pred3 = model.predict(ox2)
plt.plot(ox2,perm_pred3, color="k")
print('The coefficient of determination is {}' .format(r_sq2))

xnew=np.vstack((ox, ox1, ox2))
ynew=np.vstack((oy,oy1, oy2))
plt.scatter(xnew,ynew)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
xynew = np.hstack((xnew,ynew))
km = KMeans(
    n_clusters=3, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
)
y_km = km.fit_predict(xynew)
plt.scatter(
    xynew[y_km == 0, 0], xynew[y_km == 0, 1],
    s=50, c='lightgreen',
    marker='s', edgecolor='black',
    label='Facies: Overbanks'
)
plt.scatter(
    xynew[y_km == 1, 0], xynew[y_km == 1, 1],
    s=50, c='red',
    marker='v', edgecolor='black',
    label='Facies: Crevasse Splay'
)
plt.scatter(
    xynew[y_km == 2, 0], xynew[y_km == 2, 1],
    s=50, c='orange',
    marker='o', edgecolor='black',
    label='Facies: Channels'
)
plt.legend()
plt.title('Porosity vs. Permeability')
plt.xlabel('Porosity (%)')
plt.ylabel('Permeability (mD)')
