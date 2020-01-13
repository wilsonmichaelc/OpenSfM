import yaml
default_config_yaml = '''
# Metadata
extract_features: True          # False = load from disk

refine_with_local_map: False
tracker_lk: True
'''

def default_config():
    """Return default configuration"""
    return yaml.safe_load(default_config_yaml)