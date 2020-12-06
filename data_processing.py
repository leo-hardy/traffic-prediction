import pandas as pd
import numpy as np
import matplotlib as pt

radar = pd.read_csv('./Radar_Traffic_Counts.csv')
radar['location_name']=radar.location_name.apply(lambda x: x.strip()) 

byhour = pd.DataFrame( radar.groupby(['location_name', 'Year', 'Month', 'Day', 'Time Bin' ])['Volume'].sum() )
byhour = byhour.sort_index(level=4).sort_index(level=3).sort_index(level=2).sort_index(level=1)

my_radar = 'LAMAR BLVD / SANDRA MURAIDA WAY (Lamar Bridge)'
my_radar = pd.DataFrame( detailed.loc[ my_radar ] )
my_radar = my_radar.interpolate()