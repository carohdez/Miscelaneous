#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# Created on Fri Jun 22 17:11:27 2018

# @author: Carolina Hernandez
# 
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.learning_curve import validation_curve
from sklearn.learning_curve import learning_curve
from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

plt.style.use('seaborn-whitegrid')

# =============================================================================
# Load files and preprocessing data
# =============================================================================
#load csv files
sys.path.append("/Users/caro/Documents/MBitionTask")
path = 'Data/'
files = list()
ntrip = 0
df = pd.DataFrame()

for filename in os.listdir(path):
    files.append(filename)

for i in files:  
    dataset = pd.read_csv(path+i)
    ntrip = ntrip+1  
    dataset['Trip'] = ntrip
    df = df.append(dataset, ignore_index=True) 

#Format Time input
df['Time'] = pd.to_datetime(df['Time'].apply(lambda x: x[0:25] + 'PM' if x[25:] == 'nachm.' else x[0:25]+'AM'))

#Impute nan's 
df = df.fillna(df.mean(axis=0), axis=0)

#Rename columns
df.columns = ['Time','Latitude', 'Longitude', 'Speed', 'Acceleration','AccelerationAvg', 'Trip']

#Remove rows with no location (replace with mean would affect route plotting)
df = df.drop(df[(df.Latitude==0) | (df.Longitude==0)].index)

# =============================================================================
# Descriptive statistics
# =============================================================================
#Summary statistics
df.describe()
#Speed and acceleration per day of the week
def weekday_str(time):
    wd=time.weekday()
    if wd ==0: return "Monday"
    if wd ==1: return "Tuesday"
    if wd ==2: return "Wednesday"
    if wd ==3: return "Thursday"
    if wd ==4: return "Friday"
    if wd ==5: return "Saturday"
    if wd ==6: return "Sunday"
    
dday=df.groupby(df.Time.apply(lambda x: weekday_str(x)))
dday.describe().Speed
dday.describe().Acceleration

#Speed and acceleration  per hour of the day 
dhour=df.groupby(df.Time.apply(lambda x: x.hour))
dhour.describe().Speed
dhour.describe().Acceleration
# =============================================================================
# Plots
# =============================================================================

#Plot Histogram for Speed and Acceleration-------------------------------------
plt.hist(df[df.Speed>0].Speed);
plt.hist(df.Acceleration);
#Plot acceleration for one trip
plt.plot(df[df.Trip==1].AccelerationAvg)

#Plot max and avg velocity as function of hour of the day----------------------
fig = plt.figure()
max_speed=dhour.max().Speed
avg_speed=dhour.mean().Speed
plt.plot(max_speed.index.values, max_speed.values, label='Max. Speed')
plt.plot(avg_speed.index.values, avg_speed.values, label= 'Avg. Speed')
leg = plt.legend();
plt.title('Speed per hour of the day')
plt.xlabel('Hour')
plt.ylabel('Speed (km/h)')

#Plot max and avg velocity as function of day of the week----------------------
fig = plt.figure()
max_speed=dday.max().Speed
avg_speed=dday.mean().Speed
plt.plot(max_speed.index.values, max_speed.values, label='Max. Speed')
plt.plot(avg_speed.index.values, avg_speed.values, label= 'Avg. Speed')
leg = plt.legend();
plt.title('Speed per day of the week')
plt.xlabel('Hour')
plt.ylabel('Speed (km/h)')

#Plot Seconds of trip at different speeds--------------------------------------
dspeed=df.groupby(df.Speed)
count_speed=dspeed.Time.count()

fig = plt.figure()
plt.plot(count_speed.index.values, count_speed.values)
leg = plt.legend();
plt.title('Seconds of trip at different speeds')
plt.xlabel('Speed')
plt.ylabel('Time in total (Sec)')

#Plot progress of velocity all trips-------------------------------------------
fig = plt.figure()
for i in range(1,ntrip+1):
#for i in range(1,2):
    fig = plt.figure()
    #get time ini each trip
    ini=df[df.Trip==i].Time.min()
    plt.plot(df[df.Trip==i].Time-ini, df[df.Trip==i].Speed, label='Trip '+str(i))
    leg = plt.legend();
    

#Time on different speed ranges------------------------------------------------
def speed_range(speed):
    srange=[0,1,15,30,45,60]
    for i in range(1,len(srange)+1):
        if speed>=srange[i-1] and speed<srange[i]: return str(srange[i-1])+'-'+str(srange[i])

dspeed_range=df.groupby([df.Trip, df.Speed.apply(lambda x: speed_range(x)) ])
dspeed_range=dspeed_range.Time.count()

#Plot speed ranges in cummulative bars per trip--------------------------------
fig = plt.figure()
ax=plt.subplot(111)
l=list()
pl=list()
ranges=dspeed_range.index.levels[1].values
trips=dspeed_range.index.levels[0].values
width = 0.45

dcum=pd.Series(np.zeros(len(trips)), index=trips)
for i in ranges:
    if dcum.sum()==0 : pl.append(ax.bar(trips, dspeed_range[:, i], width))
    else : pl.append(ax.bar(trips, dspeed_range[:, i], width, bottom=dcum))
    dcum = (dcum + dspeed_range[:, i]).fillna(0)

box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
for i in range(len(pl)):
    l.append(pl[i][0])
ax.legend(reversed(l),reversed(ranges), title="Speed range (km/h)", bbox_to_anchor=(1.1, 0.5),)
plt.title('Trip times for speed ranges')
plt.xlabel('Trip')
plt.ylabel('Cumulative Time (Sec)')

#Plot speed ranges in cummulative bars per hour of the day --------------------
dspeed_range=df.groupby([df.Time.apply(lambda x: x.hour), df.Speed.apply(lambda x: speed_range(x)) ])
dspeed_range=dspeed_range.Time.count()
ranges=dspeed_range.index.levels[1].values
hours=dspeed_range.index.levels[0].values

#mock hours of the day missing from data for plotting purposes (set speed=0)
for i in hours:
    for j in ranges:
        try : dspeed_range[(i, j)]
        except KeyError: 
            s = pd.Series(0, index=pd.MultiIndex.from_product([[i], [j]]))
            dspeed_range=dspeed_range.append(s)
dspeed_range=dspeed_range.sort_index()     

#Plot
fig = plt.figure()
ax=plt.subplot(111)
l=list()
pl=list()
width = 0.45
dcum=pd.Series(np.zeros(len(hours)), index=hours)
i=ranges[0]
for i in ranges:
    if dcum.sum()==0 : pl.append(ax.bar(hours, dspeed_range[:, i], width))
    else : pl.append(ax.bar(hours, dspeed_range[:, i], width, bottom=dcum))
    dcum = (dcum + dspeed_range[:, i]).fillna(0)

box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
for i in range(len(pl)):
    l.append(pl[i][0])
ax.legend(reversed(l),reversed(ranges), title="Speed range (km/h)", bbox_to_anchor=(1.1, 0.5),)
plt.title('Trip times for speed ranges per hour of the day')
plt.xlabel('Hour of the day')
plt.ylabel('Cumulative Time (Sec)')
  


#Plot routes of each trip------------------------------------------------------
#Main Berlin Mitte spots
spots= {'Spot': ['Kreuzberg','Bergmannkiez','Alexanderplatz','Potsdamer_Platz','U-Stadtmitte','U-Rathaus-Neukölln', 'U-Hermannplatz'],
        'Latitude': [52.4966209,52.489692,52.5219216,52.5099204,52.5122237,52.4815142,52.4865993 ],
        'Longitude': [13.3758185,13.3870661,13.411026,13.3778095,13.3903756,13.4329674 ,13.422284]}

spots=pd.DataFrame(spots)
fig, ax = plt.subplots()
l=list()
pl=list()
plt.plot(spots.Longitude, spots.Latitude , '.', color='gray');

for i in trips:
    pl.append(plt.plot(df[df.Trip==i].Longitude, df[df.Trip==i].Latitude))
    ini=df[df.Trip==i].index.values.min()
    end=df[df.Trip==i].index.values.max()
    plt.plot(df.loc[ini].Longitude,df.loc[ini].Latitude, '.', color='black');
    plt.plot(df.loc[end].Longitude,df.loc[end].Latitude, '*', color='black');

style = dict(size=10, color='gray')
for i in range(len(spots)):
    ax.text(spots.Longitude[i],spots.Latitude[i],  spots.Spot[i], **style, horizontalalignment='center')

for i in range(len(pl)):
    l.append(pl[i][0])
ax.legend(l,trips, title ="Trip", bbox_to_anchor=(1.2, 0.7),)
#legend1=ax.legend(["Ini"], bbox_to_anchor=(1.2, 0.2),)
#legend2=ax.legend(["* End"], bbox_to_anchor=(1.2, 0.1),)
#plt.gca().add_artist(legend1)
#plt.gca().add_artist(legend2)

#ax.legend(["a simple line"], bbox_to_anchor=(1.0, 0.7),)
plt.title('Route for each trip')
plt.xlim(df.Longitude.min()-0.02,df.Longitude.max()+0.005)
plt.xlabel('Longitude (Degrees)')
plt.ylabel('Latitude (Degrees)')

# =============================================================================
# Regresion
# =============================================================================
#Predict the max speed given the hour of the day

# =============================================================================
# 0. plot data to asses linearity
# 1. choose the model
# 2. choose parameters
# 3. arrange data into X and Y
# 4. Fit model to the data
# 5. Apply the Model to new data
# 6. Plot fit
# 7. validate model
# =============================================================================

# 0. plot data to asses linearity
#Speed per hour of the day 
dquarter=df.groupby(df.Time.apply(lambda x: x.minute))
dquarter.describe().Speed

fig = plt.figure()
max_speed=dquarter.max().Speed
#max_speed=max_speed.drop(max_speed[max_speed==0].index) #drop hours where max speed is 0 (special case here due to the small size of dataset)
avg_speed=dquarter.mean().Speed
plt.plot(max_speed.index.values, max_speed.values, label='Max. Speed')
plt.plot(avg_speed.index.values, avg_speed.values, label= 'Avg. Speed')
plt.scatter(max_speed.index.values, max_speed.values);
plt.scatter(avg_speed.index.values, avg_speed.values);
leg = plt.legend();
plt.title('Speed per minute of the hour (*)')
plt.xlabel('Minute')
plt.ylabel('Speed (km/h)')

# 1. choose the model
# 2. choose parameters

#since non-linearity, assess polinomial features
model = make_pipeline(PolynomialFeatures(7),LinearRegression())
# the number of degrees was adjusted after the validation curve assesmsent


# 3. arrange data into X and Y
X=max_speed.index.values # Feature matrix: unit of time
y=max_speed.values # Target value: speed

Xfit = np.linspace(0, 59, 46, dtype=int) #new values to predict

# 4. Fit model to the data
model.fit(X[:, np.newaxis], y)

# 5. Apply the Model to new data
yfit = model.predict(Xfit[:, np.newaxis])

# 6. Plot fit
#plt.scatter(X, y)
plt.plot(Xfit, yfit);

# 7. validate model

# Validate accuracy

# split the data: 90% training and 10% validation 
X1, X2, y1, y2 = train_test_split(X, y, random_state=0,train_size=0.9)
# fit the model on one set of data
model.fit(X1.reshape(-1, 1), y1)

# evaluate the model on the second set of data
y2_pred = model.predict(X2.reshape(-1, 1))
r2=r2_score(y2, y2_pred)  
print("The coefficient of determination (R2):", r2)
if r2<2 : print("The coefficient is negative. Try a better model, or try to increase the training set!!")
else: print("The model seems to represent a good fit!")

# validation curve, to assess the adequate degrees for polynomial model
fig = plt.figure()
degree = np.arange(0, 20)
train_score, val_score = validation_curve(model, X.reshape(-1, 1), y,'polynomialfeatures__degree', degree, cv=7)
# cv correspond to the number of folds in cross-validation

plt.plot(degree, np.median(train_score, 1), color='blue', label='training score')
plt.plot(degree, np.median(val_score, 1), color='red', label='validation score')
plt.legend(loc='best')
#plt.ylim(0, 1)
plt.xlabel('degree')
plt.ylabel('score');

# =============================================================================
# Clustering
# =============================================================================
# Identify clusters of areas with slower avg speed for each trip

nclusters=4
ntrips=df.Trip.max()
for ntrip in range(1,ntrips+1):
    lat=df.Latitude[df.Trip==ntrip]
    lon=df.Longitude[df.Trip==ntrip]
    spe=df.Speed[df.Trip==ntrip]
    X= np.vstack((lon,lat,spe)).T
    nticks=np.linspace(0, np.array(spe).max(), nclusters, dtype=int)
    
    #Get kmeans
    kmeans = KMeans(n_clusters=nclusters)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    
    #Plot routs and clusters
    fig,ax = plt.subplots()
    
    plt.plot(spots.Longitude, spots.Latitude , '.', color='gray');
    ini=df[df.Trip==ntrip].index.values.min()
    end=df[df.Trip==ntrip].index.values.max()
    plt.plot(df.loc[ini].Longitude,df.loc[ini].Latitude, '.', color='black');
    plt.plot(df.loc[end].Longitude,df.loc[end].Latitude, '*', color='black');

    style = dict(size=10, color='gray')
    for i in range(len(spots)):
        ax.text(spots.Longitude[i],spots.Latitude[i],  spots.Spot[i], **style, horizontalalignment='center')


    pl=plt.scatter(X[:, 0], X[:, 1], c=y_kmeans,  cmap='viridis')
    tlabels=np.array(range(nclusters))
    cb = plt.colorbar(pl, label='Speed (km/h)', ticks=tlabels)
    cb.ax.set_yticklabels(nticks)
    plt.xlim(df.Longitude.min()-0.02,df.Longitude.max()+0.005)
    plt.title("Average speed per zone - Trip "+ str(ntrip))
    plt.xlabel('Longitude (Degrees)')
    plt.ylabel('Latitude  (Degrees)')



# Compare with real speed ranges during route
#Time on different speed ranges------------------------------------------------
ntrip=1
def speed_range_int(speed):
    srange=[0,1,20,40,60]
    for i in range(len(srange)):
        if speed>=srange[i-1] and speed<srange[i]: return i-1
fig = plt.figure()
pl=plt.scatter(X[:, 0], X[:, 1], c=pd.Series(X[:, 2]).apply(lambda x: speed_range_int(x)),  cmap='viridis')
tlabels=np.array(range(nclusters))
cb = plt.colorbar(pl, label='Speed', ticks=tlabels)
cb.ax.set_yticklabels(nticks)
plt.title("Real speed")




