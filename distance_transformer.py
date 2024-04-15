
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class DistanceToCenterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, central_lon, central_lat):
        self.central_lon = central_lon
        self.central_lat = central_lat

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Define haversine formula to calculate distances
        def haversine(lon1, lat1, lon2, lat2):
            # Convert decimal degrees to radians
            lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
            # Haversine formula
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            r = 6371  # Radius of Earth in kilometers
            return c * r

        # Apply distance calculation
        distances = X.apply(
            lambda row: haversine(row['longitude'], row['latitude'], self.central_lon, self.central_lat),
            axis=1
        )
        return np.array(distances).reshape(-1, 1)