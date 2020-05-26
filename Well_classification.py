

# Classification of Well Functionalities in Tanzania

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm


sys.path.append('C:\\Users\\alex\\Documents\\myPython\\git\\wells\\')

#import specific functions
sys.path.append('well_functions')
import well_functions as wf



#Import data - due to a persistant error, you will have to change the locations below vvvv
trainvals = pd.read_csv('C:/Users/alex/Documents/myPython/git/wells/data/training_values.csv')
trainlabs = pd.read_csv('C:/Users/alex/Documents/myPython/git/wells/data/training_labels.csv')
testvals = pd.read_csv('C:/Users/alex/Documents/myPython/git/wells/data/test_values.csv')
cities = pd.read_csv('C:/Users/alex/Documents/myPython/git/wells/data/cities.csv')

# possibly to this vvv
#trainvals = pd.read_csv('/data/training_values.csv')
#trainlabs = pd.read_csv('/data/training_labels.csv')
#testvals = pd.read_csv('/data/test_values.csv')
#cities = pd.read_csv('/data/cities.csv')


#%%

# merging data

testvals['test'] = 1
trainvals['test'] = 0

df = pd.concat([trainvals,testvals])
df.index = range(0,len(df.iloc[:,0]))

#%%
# A look at the datatpype for each column vvv
types = []
for i in range(0,41):
    types.append(type(df.iloc[1,i]))
    
types = np.array(types)

dtypes = pd.DataFrame({'Column name' : df.columns, f'Data type' : types})

print(dtypes)
print()
print()



# A look at the number of null values in each column vvv
nulls = df.isnull().sum()

nindex = []
for i in range(0,40):
    if nulls[i] != 0:
        nindex.append(i)
        
nullbool = np.array((0),dtype = bool)

for i in range(0,40):
    if i+1 in nindex:
        nullbool = np.append(nullbool,[True])
    else:
        nullbool = np.append(nullbool,[False])
        
nulldf = pd.DataFrame({'# Null Values' :nulls[nullbool] })
print(nulldf)       
        
# Here you can see that some of the features have thousands of null values,
# getting rid of the rows with just 1 null value would be 
# ridiculous as this would remove over half of our training data.



#%%

# THIS CELL IS OPTIONAL AND TAKES ABOUT A FEW MINUTES TO RUN #

# checking possible invalid terms in string columns 

# It turns out only 0 and unknown are worth looking into.

# =============================================================================
# check = ['0','unknown']# 'invalid', 'NaN', 'error', 'Null' ]
# p = 0
# 
# listofdf = []
# for k in tqdm(check):
#     counts =[]
#     indexx = []
#     for i in range(0,len(df.iloc[0,:])):
#         count = 0
#         if type(df.iloc[2,i]) == str and type(df.iloc[3,i]) == str:
#             indexx = np.append(indexx,i)
#             for j in range(0,len(df.iloc[:,0])):
#                 if type(df.iloc[j,i]) == str:
#                     if df.iloc[j,i].count(k) > 0:
#                         count = count + 1
#         counts = np.append(counts,count)
#         
#     tindex = np.array((0),dtype = bool)
#     for i in range(0,40):
#         if i in indexx and counts[i+1] != 0:  
#             tindex = np.append(tindex,[True])
#         else:
#             tindex  = np.append(tindex,[False])
#     
#     cols = []
#     for i in range (0,40):
#         cols = np.append(cols,df.columns[i])
#         
#     countss = pd.DataFrame({'Column name' : df.columns[tindex], f'Number of \'{check[p]}\'s' :counts[tindex] })
#     listofdf.append(countss)
#     p=p+1
# 
# for i in range(len(listofdf)):
#     print()
#     print(listofdf[i])
# =============================================================================

#%%
    
# Checking the '0's in the extraction_type_group feature.
    #(only works if above cell has been ran)
# =============================================================================
# yes = 0
# for i in range(0, len(df.iloc[:,0])):
#     if df['extraction_type_group'][i] == 'swn 80': #swn 80 is known to be valid
#         yes = yes + 1
# 
# if yes == listofdf[0].iloc[2,1]:
#     print()
#     print('The \'0\'s in',listofdf[0].iloc[2,0],'are valid')
# =============================================================================
    
# The '0's in the features 'funder' and 'subvillage' are in fact invalid and should
# be dealt with.
    
# Many of the features have an abundance of 'unknown' values which are also yet to be
# dealt with. Im not goint to delete the rows, but modify the 'unknown' values 
# and create a dummy variable for when it was modified, because the fact
# that the data is unknown to us, is still useful information.
    
#%%

# Parsing Data
    
# First I have dropped the scheme_name column as it contained far too many null values,
# then I have dropped rows with null values and non-sensical values.

#indexing the null values
    
index0 = df['construction_year'] == 0
index1 = df['longitude'] == 0
index2 = df['latitude'] == -2*10**(-8)
index3 = df['gps_height'] == 0
index4 = df['scheme_name'] == df['scheme_name']
index5 = df['subvillage'] == np.nan 
index6 = df['subvillage'] == 0

#creating dummy variables to track unknowns in data

df['dummy_construction_year'] = index0.replace(True,1).replace(False,0)
df['dummy_longitude'] = index1.replace(True,1).replace(False,0)  
df['dummy_latitude'] = index2.replace(True,1).replace(False,0)
df['dummy_gps_height'] = index3.replace(True,1).replace(False,0)
df['dummy_scheme_name'] = index4.replace(True,1).replace(False,0)
df['dummy_subvillage'] = index5.replace(True,1).replace(False,0)
df['dummy_subvillage1'] = index6.replace(True,1).replace(False,0)

#replacing null values - this should be automated in the future

df['construction_year'] = df['construction_year'].replace(0,df['construction_year'].replace(0,np.nan).dropna().mean())
df['longitude'] = df['longitude'].replace(0,df['longitude'].replace(0,np.nan).dropna().mean())
df['latitude'] = df['latitude'].replace(-2*10**(-8),df['latitude'].replace(0,np.nan).dropna().mean())
df['gps_height'] = df['gps_height'].replace(0,df['gps_height'].replace(0,np.nan).dropna().mean())
df['funder'] = df['funder'].replace(np.nan,"unknown")
df['funder'] = df['funder'].replace("0","unknown",regex=True)
df['installer'] = df['installer'].replace(np.nan,"unknown")
df['public_meeting'] = df['public_meeting'].replace(np.nan,"unknown")
df['scheme_management'] = df['scheme_management'].replace(np.nan,"unknown")
df['scheme_name'] = df['scheme_name'].replace(np.nan,"unknown")
df['permit'] = df['permit'].replace(np.nan,"unknown")

df = df.drop(columns = ['subvillage'])


#%%

#plotting the wells

#takes about 20 seconds but worth it
#color = height!

dftrain = df[df['test'] == 0]
dftrain['status_group'] = trainlabs['status_group']
wf.plot_wells(dftrain,cities)

#%%
# Extracting information - average number of well users per well per city etc.

usermean,d,dmin,dmin1 = wf.extract(df,cities)
usermean4,d4,dmin4,dmin14 = wf.extract(dftrain,cities)
cities['usermean'] = usermean4.iloc[:,1]
print()
print(usermean4)


#%%

# # Plotting the mean number of users against the population of each city.
# Fitting a function to the data
# not the most beautiful visualisation

plt.figure()

coeffs = np.polyfit(cities['population'],cities['usermean'],2)
x = np.linspace(0,cities['population'].max(),100)
y = np.polyval(coeffs,x)
plt.plot(x,y,color = "k", linestyle = "--", label = " Fitted Function")
plt.ylim(top = 600, bottom = 0)
    
plt.scatter(cities['population'],cities['usermean'], label = "Data")

plt.ylabel("User Mean")
plt.xlabel("Population")
plt.title("User Mean vs Population")
plt.legend()
plt.show()


#%%

#Summarising Data with Aggregation and Grouping

agg = wf.aggregate(dftrain,case = 1)
print()
print(agg)
agg = wf.aggregate(dftrain,case = 2)
print()
print(agg)

#%%

#preparing for classification

#first lets modify our feature data
dfX = df.drop(columns = ['xcoord','ycoord','region_code','recorded_by','wpt_name','date_recorded','funder','installer','ward','scheme_name'])
dfX = dfX[df['test'] == 0]
#adding extracted data 
dfX['dmin'] = dmin
dfX['dmin1'] = dmin1  
for i in range(0,10):
    dfX[f'd{i}'] = d.iloc[:,i]

xtypes = []
strcols = []
xtypesi = []

#sorting out string data

dfX['permit'] = dfX['permit'].astype(str)
dfX['public_meeting'] = dfX['public_meeting'].astype(str)

for i in range(0,len(dfX.iloc[0,:])):
    if type(dfX.iloc[1,i]) == str:
        xtypes.append(type(dfX.iloc[1,i]))
        xtypesi.append(i)
        strcols.append(dfX.columns[i])
        
xtypes = np.array(xtypes)
xtypes = pd.DataFrame({'Column name' : strcols, f'Data type' : xtypes, 'Column #' : xtypesi})

# enumerating - takes ages
for i in range(0,len(xtypesi)):
    for n, m in enumerate(list(dfX[f'{dfX.columns[xtypesi[i]]}'].unique())):
        dfX[f'{dfX.columns[xtypesi[i]]}'] = dfX[f'{dfX.columns[xtypesi[i]]}'].replace({m: n})       

                      
#%%

#converting date_recorded into floats and adding to features
datearray = np.array(pd.to_datetime(df['date_recorded'][df['test'] == 0])).astype(float)
dfX['date_recorded'] = pd.DataFrame(datearray).set_index(dfX.index)


#%%

# dummy variables for catagorical data
dfX = pd.get_dummies(dfX, columns=[  'basin',
                                     'region',
                                     'lga',
                                     'scheme_management',
                                     'extraction_type',
                                     'extraction_type_group',
                                     'extraction_type_class',
                                     'management',
                                     'management_group',
                                     'permit',
                                     'public_meeting',
                                     'payment',
                                     'payment_type',
                                     'quantity',
                                     'quantity_group',
                                     'dmin',
                                     'district_code',
                                     'num_private',
                                     'water_quality',
                                     'quality_group',
                                     'source',
                                     'source_type',
                                     'source_class',
                                     'waterpoint_type',
                                     'waterpoint_type_group'])

#%%

y = trainlabs['status_group']
for n, m in enumerate(list(y.unique())):
    y=y.replace({m: n})       
    
#%%

#This cell was simply to test which max_depth gave the highest test accuracy
    
# =============================================================================
# X_train, X_test, y_train, y_test = train_test_split(dfX, y, stratify=y, random_state=42, test_size = 0.25)
# 
# scores = []
# scoress = []
# 
# for i in np.arange(10,45,5):
#     rf = RandomForestClassifier(random_state=1, max_depth = i, n_estimators = 10)
#     rf.fit(X_train, y_train)
#     scores = np.append(scores, rf.score(X_train, y_train))
#     scoress = np.append(scoress, rf.score(X_test, y_test))
# 
# plt.figure()
# plt.plot(range(0,len(scores)),scores, label = "Training Accuracy")
# plt.plot(range(0,len(scoress)),scoress, label = "Test Accuracy")
# plt.xlabel("Max depth")
# plt.ylabel("Accuracy")
# plt.legend()
# #plt.savefig('depthplot.pdf')
# plt.show()
# 
# =============================================================================

#%%
    
#deploying random forest

X_train, X_test, y_train, y_test = train_test_split(dfX, y, stratify=y, random_state=42, test_size = 0.25)

rf = RandomForestClassifier(random_state=1, max_depth = 20, n_estimators = 10)
rf.fit(X_train, y_train)
print(rf.score(X_test, y_test))








