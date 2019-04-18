import constants
import numpy as np

def distance(lat1, lat2, lon1,lon2):
    
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))


def isAirport(latitude,longitude,airport_name='JFK'):
    
    if latitude>=constants.nyc_airports[airport_name]['min_lat'] and latitude<=constants.nyc_airports[airport_name]['max_lat'] and longitude>=constants.nyc_airports[airport_name]['min_lng'] and longitude<=constants.nyc_airports[airport_name]['max_lng']:
        return 1
    else:
        return 0
    
def isLowerManhattan(lat,lng):
    if lat>=constants.lower_manhattan_boundary['min_lat'] and lat<=constants.lower_manhattan_boundary['max_lat'] and lng>=constants.lower_manhattan_boundary['min_lng'] and lng<=constants.lower_manhattan_boundary['max_lng']:
        return 1
    else:
        return 0
    
def getBorough(lat,lng):
    
    locs=constants.nyc_boroughs.keys()
    for loc in locs:
        if lat>=constants.nyc_boroughs[loc]['min_lat'] and lat<=constants.nyc_boroughs[loc]['max_lat'] and lng>=constants.nyc_boroughs[loc]['min_lng'] and lng<=constants.nyc_boroughs[loc]['max_lng']:
            return loc
    return 'others'    