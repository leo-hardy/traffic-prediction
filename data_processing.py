import pandas as pd
import numpy as np
import matplotlib as pt

radar = pd.read_csv('../Radar_Traffic_Counts.csv')
radar['location_name']=radar.location_name.apply(lambda x: x.strip())

# compute volume per hour
byhour = pd.DataFrame( radar.groupby(['location_name', 'Year', 'Month', 'Day', 'Time Bin' ])['Volume'].sum() )
byhour = byhour.sort_index(level=4).sort_index(level=3).sort_index(level=2).sort_index(level=1)
byhour = byhour.unstack(['Year', 'Month', 'Day', 'Time Bin'])

# we choose LAMAR BLVD because it is the most complete dataset (few Nan)
my_radar = 'LAMAR BLVD / SANDRA MURAIDA WAY (Lamar Bridge)'
#my_radar = '700 BLK E CESAR CHAVEZ ST'
my_radar = pd.DataFrame( byhour.loc[ my_radar ] )
my_radar = my_radar.interpolate().clip(0, 1000) # mean + 2*std ~~ 1000

my_radar.to_csv( '../LAMAR BLVD'+'.csv' )
#my_radar.to_csv( '../CESAR CHAVEZ'+'.csv' )
