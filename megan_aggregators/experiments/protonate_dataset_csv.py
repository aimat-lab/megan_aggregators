import os
import csv
import pathlib

PATH = pathlib.Path(__file__).parent.absolute()
agg_path = os.path.join(PATH, 'assets', 'cleaned_data_aggregators_charged.csv')
nonagg_path = os.path.join(PATH, 'assets', 'cleaned_data_nonaggregators_charged.csv')

data = []

index_unique = 0
index = 0
for label, path in {True: agg_path, False: nonagg_path}.items():
    with open(path, mode='r') as file:
        lines = file.readlines()
        for line in lines:
            smiles = line.split(',')
            num_elements = len(smiles)
            
            if num_elements > 1:
                
                for value in smiles[1:]:
                    value = value.replace('\n', '').replace(' ', '')
                    data.append({
                        'index': index,
                        'unique': index_unique,
                        'smiles': value,
                        'aggregator': int(label),
                        'nonaggregator': int(not label),
                    })
                    index += 1
                
            index_unique += 1
    

dest_path = os.path.join(PATH, 'assets', 'aggregators_binary_protonated.csv')
with open(dest_path, mode='w') as file:
    writer = csv.DictWriter(file, fieldnames=['index', 'unique', 'smiles', 'aggregator', 'nonaggregator'])
    writer.writeheader()
    
    for d in data:
        writer.writerow(d)
