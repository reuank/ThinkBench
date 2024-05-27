import json

from numpy import float32


class TotalResultEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float32):
            return float(obj)
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        else:
            return obj.__dict__


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float32):
            return float(obj)
        else:
            return obj.__dict__
