fig1 = plt.subplots(figsize=(7,10))

ax1 = plt.subplot2grid((1,1), (0,0), rowspan=1, colspan=1)
ax2 = ax1.twiny()

ax1.plot('CNPOR', 'Depth', data=df_filtered, color='red', lw=1.5)
ax1.set_xlim(45, -15)
ax1.set_xlabel('CNPOR')
ax1.xaxis.label.set_color("red")
ax1.tick_params(axis='x', colors="red")
ax1.spines["top"].set_edgecolor("red")

ax2.plot('RHOB', 'Depth', data=df_filtered, color='yellow', lw=1.5)
# ax2.set_xlim(45, -15)
ax2.set_xlabel('RHOB')
ax2.xaxis.label.set_color("yellow")
ax2.spines["top"].set_position(("axes", 1.08))
ax2.tick_params(axis='x', colors="yellow")
ax2.spines["top"].set_edgecolor("yellow")

x1=df_filtered['CNPOR']
x2=df_filtered['RHOB']

x = np.array(ax1.get_xlim())
z = np.array(ax2.get_xlim())

# here we add this equation to find the difference of x-values of each graph
nz=((x2-np.max(z))/(np.min(z)-np.max(z)))*(np.max(x)-np.min(x))+np.min(x)

# here we fill this difference with the colors (green when RHOB on the right and CNPOR on the left, yellow otherwise)
ax1.fill_betweenx(df_filtered['Depth'], x1, nz, where=x1>=nz, interpolate=True, color='green')
ax1.fill_betweenx(df_filtered['Depth'], x1, nz, where=x1<=nz, interpolate=True, color='yellow')

for ax in [ax1, ax2]:
    ax.set_ylim(2900, 2400)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")


fig2 = plt.subplots(figsize=(7,10)) # subplots same as before

ax1 = plt.subplot2grid((1,1), (0,0), rowspan=1, colspan=1) # using subplot2grid to put the two figures in 1 graph
ax2 = ax1.twiny() # twiny for the same purpose

ax1.plot('RILD', 'Depth', data=df_filtered, color='black', lw=1.5) # plot CNPOR from dt_filtered, color red, and the line width is 1.5
# ax1.set_xlim(45, -15) # set the ranges from 45 to -15 (to invert the range)
ax1.set_xlabel('RILD') # label x-axis
ax1.xaxis.label.set_color("black") # x-axis red color
ax1.tick_params(axis='x', colors="black") # parameters of x-axis red color
ax1.spines["top"].set_edgecolor("black") # set the x-axis from the top with edge color red

ax2.plot('RLL3', 'Depth', data=df_filtered, color='blue', lw=1.5) # plot RHOB same as CNPOR
ax2.set_xlabel('RLL3')
ax2.xaxis.label.set_color("blue")
ax2.spines["top"].set_position(("axes", 1.08))
ax2.tick_params(axis='x', colors="blue")
ax2.spines["top"].set_edgecolor("blue")

# here we insert a for loop for both axis to set the range of the y-axis
for ax in [ax1, ax2]:
    ax.set_ylim(2900, 2400)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

df_filtered["DEPTHM"]=df_filtered['Depth']*0.3048 
ax = df_filtered.plot(x='GR', y='DEPTHM', c='black', lw=0.5, legend=False, figsize=(7,10))
logi=df_filtered["GR"] <50
f=logi.sum()
z=0.5*0.3048 
thickness=z*f

fig3, ax = plt.subplots(figsize=(25,10))
#Set up the plot axes
ax1 = plt.subplot2grid((1,6), (0,0), rowspan=1, colspan = 1)
ax2 = plt.subplot2grid((1,6), (0,1), rowspan=1, colspan = 1, sharey = ax1)
ax3 = plt.subplot2grid((1,6), (0,2), rowspan=1, colspan = 1, sharey = ax1)
ax5 = ax3.twiny() #Twins the y-axis for the CNPOR with RHOB
ax7 = ax2.twiny()

# As our curve scales will be detached from the top of the track,
# this code adds the top border back in without dealing with splines
ax10 = ax1.twiny()
ax10.xaxis.set_visible(False)
ax11 = ax2.twiny()
ax11.xaxis.set_visible(False)
ax12 = ax3.twiny()
ax12.xaxis.set_visible(False)

# Gamma Ray track
ax1.plot('GR', 'Depth', data=df_filtered, color = "green", lw = 0.5)
ax1.set_xlabel("Gamma Ray")
ax1.xaxis.label.set_color("green")
# ax1.set_xlim(0, 200)
ax1.set_ylabel("Depth (m)")
ax1.tick_params(axis='x', colors="green")
ax1.spines["top"].set_edgecolor("green")
ax1.title.set_color('green')
# ax1.set_xticks([0, 50, 100, 150, 200])


# RILD
ax2.plot('RILD', 'Depth', data=df_filtered, color='black', lw=1.5) # plot CNPOR from dt_filtered, color red, and the line width is 1.5
# ax2.set_xlim(45, -15) # set the ranges from 45 to -15 (to invert the range)
ax2.set_xlabel('RILD') # label x-axis
ax2.xaxis.label.set_color("black") # x-axis red color
ax2.tick_params(axis='x', colors="black") # parameters of x-axis red color
ax2.spines["top"].set_edgecolor("black") # set the x-axis from the top with edge color red
ax2.title.set_color('black')

# RLL3
ax7.plot('RLL3', 'Depth', data=df_filtered, color='blue', lw=1.5) # plot RHOB same as CNPOR
ax7.set_xlabel('RLL3')
ax7.xaxis.label.set_color("blue")
ax7.spines["top"].set_position(("axes", 1.08))
ax7.tick_params(axis='x', colors="blue")
ax7.spines["top"].set_edgecolor("blue")
ax7.title.set_color('blue')

# CNPOR
ax3.plot('CNPOR', 'Depth', data=df_filtered, color='red', lw=1.5)
ax3.set_xlim(45, -15)
ax3.set_xlabel('CNPOR')
ax3.xaxis.label.set_color("red")
ax3.tick_params(axis='x', colors="red")
ax3.spines["top"].set_edgecolor("red")

# RHOB
ax5.plot('RHOB', 'Depth', data=df_filtered, color='yellow', lw=1.5)
# ax2.set_xlim(45, -15)
ax5.set_xlabel('RHOB')
ax5.xaxis.label.set_color("yellow")
ax5.spines["top"].set_position(("axes", 1.08))
ax5.tick_params(axis='x', colors="yellow")
ax5.spines["top"].set_edgecolor("yellow")

x3=df_filtered['CNPOR']
x5=df_filtered['RHOB']

x = np.array(ax3.get_xlim())
z = np.array(ax5.get_xlim())

# here we add this equation to find the difference of x-values of each graph
nz=((x5-np.max(z))/(np.min(z)-np.max(z)))*(np.max(x)-np.min(x))+np.min(x)

# here we fill this difference with the colors (green when RHOB on the right and CNPOR on the left, yellow otherwise)
ax3.fill_betweenx(df_filtered['Depth'], x3, nz, where=x3>=nz, interpolate=True, color='green')
ax3.fill_betweenx(df_filtered['Depth'], x3, nz, where=x3<=nz, interpolate=True, color='yellow')

# Common functions for setting up the plot can be extracted into
# a for loop. This saves repeating code.
for ax in [ax1, ax2, ax3]:
    ax.set_ylim(2900, 2400)
    ax.grid(which='major', color='lightgrey', linestyle='-')
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.spines["top"].set_position(("axes", 1.02))

for ax in [ax2, ax3]:
    plt.setp(ax.get_yticklabels(), visible = False)

plt.tight_layout()
# fig.subplots_adjust(wspace = 0.15)
plt.show()

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
