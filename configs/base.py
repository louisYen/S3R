''' This config file will handle the video anomaly detection'''

from pathlib import Path

data_root = Path('data')
dictionary_root = Path('dictionary')
quantize_size = 32
backbone = 'i3d'


