# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 00:19:02 2020

@author: alex
"""


def plot_wells(df,cities):
    
    # =============================================================================
    # This function returns a plot for the cities and wells, where the size of the 
    # marker indicated the users/population for the wells/cities and the colour 
    # represents the altitude of the well.
    # =============================================================================

    from matplotlib import pyplot as plt
    import  numpy as np
    latcit = cities['latitude']
    loncit = cities['longitude']
    popcit = cities['population']
    namecit = cities['city']
    
    
    lat = df['latitude']
    lon = df['longitude']
    pop = df['users']
    gps_height = df['gps_height']
    gps_heightlin = gps_height * 7.059654*10**(-4)-0.95552
    
    plt.figure(figsize = (8,6))
   #hindex = gps_heightlin.values == -0.23623950041624153
    hindex = gps_heightlin.values == -0.2375738002224398
    nhindex = np.invert(hindex)
    plt.scatter(lon[nhindex],lat[nhindex],s = pop[nhindex]*(0.01)+2, c = gps_heightlin[nhindex], label = "Wells",alpha = 0.7)
    plt.scatter(lon.values[hindex],lat.values[hindex],s = pop.values[hindex]*(0.01)+2, c = 'orange', label = "Unknown GPS Height",alpha = 0.6)
    plt.scatter(loncit,latcit,s = popcit*(4*10**(-5))+40, color = (0,0,0), label = "Cities")
    for i in range(0,10):
        plt.annotate(str(namecit[i]), xy = (loncit[i]+0.23,latcit[i] + 0.1), fontsize=11)
        
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Exercise 3b : The Distribution of Wells in Tanzania")
    plt.legend()
    plt.savefig('exercise3aplot.pdf')
    plt.show()
    
 
    
def extract(df,cities):
    
    # =============================================================================
    # This function returns the average number of users for wells
    # within 200km of each city.
    # =============================================================================
    
    import pandas as pd
    import numpy as np
    df['xcoord'] = (40008000/360)*df['latitude']
    df['ycoord'] = (40075160/360)*df['longitude']*np.cos(df['latitude']*2*np.pi/360)

    cities['xcoord']= (40008000/360)*cities['latitude']
    cities['ycoord'] = (40075160/360)*cities['longitude']*np.cos(cities['latitude']*2*np.pi/360)

    d = pd.DataFrame(np.zeros((len(df.iloc[:,0]),10)),index = df.index.values)
    for i in range(0,10):
         d.iloc[:,i] = np.sqrt((df['xcoord']-cities['xcoord'][i])**2 + (df['ycoord']-cities['ycoord'][i])**2)
         
    dmin = d.idxmin(axis=1)
    
    usermean = np.zeros((10))
    for i in range(0,10):
        dbool = d.iloc[:,i] < 200000
        closewell = d.iloc[:,i][dbool]
        usermean[i] = df.loc[closewell.index.values]['users'].mean()
    
    
    usermean = pd.DataFrame({'City name' : cities['city'], 'Average # of Users' : usermean })
    
    dmin1 = usermean.iloc[dmin.iloc[:],1]  
    dmin1 = pd.DataFrame(dmin1).set_index(dmin.index)
    
    return usermean,d,dmin,dmin1
    
    
    
def aggregate(df,case):
        
    # =============================================================================
    # This function returns the mode decade of construction for each type of extraction, 
    # as well as the mode of the three possible well conditions, depending on the case.
    # =============================================================================
    
    import pandas as pd
    import numpy as np
    
    if case == 1:
        df['decade'] = df['construction_year'].apply(lambda x:10*np.floor(x/10))
        dftest = df.groupby(['extraction_type','decade']).count()
        
        dft4 = pd.DataFrame(np.zeros((18,6)), index = df['extraction_type'].drop_duplicates(), columns = df['decade'].drop_duplicates().sort_values().astype(int))

        for i,j in dftest.index:
            for k in dft4.columns:
                for l in dft4.index:
                    if j == k and i == l:
                       dft4.loc[l,k] = dftest.loc[i,j].values[0]
                       
        dft4 = dft4.astype(int)
        
    if case == 2:
        
        df['decade'] = df['construction_year'].apply(lambda x:10*np.floor(x/10))
        dftest = df.groupby(['decade','status_group']).count()     
        
        dft = []
        dft1 = []
        
        for i,j in dftest.index:
             dft = np.append(dft,dftest.loc[i]['id'].idxmax())
             dft1 = np.append(dft1,i)
         
        dft2 = pd.DataFrame(dft, columns = ['Mode Status'], index = dft1.astype(int))
        dft4 = dft2.loc[~dft2.index.duplicated(keep='first')]
        dft4.index.name = 'Decade'
        
    return dft4












