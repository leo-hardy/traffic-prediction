import pandas as pd
import numpy as np

radar = pd.read_csv('./Radar_Traffic_Counts.csv')
radar.head()
radar.describe()


byhour = pd.DataFrame( radar.groupby(['Year', 'Month', 'Day', 'Hour', 'Minute', 'location_latitude', 'location_longitude'])['Volume'].sum() )
byhour.groupby( level=['Year', 'Month', 'Day', 'Hour', 'Minute'] ).count().describe()