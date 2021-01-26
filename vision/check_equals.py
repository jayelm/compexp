"""
Post-hoc results
"""


from loader.data_loader import formula as F
import pandas as pd
from collections import Counter
from pyeda.boolalg import expr
from tqdm import tqdm

EXPTS = {
    1: 'result/resnet18_places365_broden_ade20k_neuron_1/tally.csv',
    3: 'result/resnet18_places365_broden_ade20k_neuron_3/tally.csv',
    5: 'result/resnet18_places365_broden_ade20k_neuron_5/tally_layer4_0_127.csv',
    10: 'result/resnet18_places365_broden_ade20k_neuron_10_nopenalty/tally_layer4.csv',
}

UNIT_RANGE = None

csvs = {}
for length, csvname in EXPTS.items():
    records = pd.read_csv(csvname).to_dict('records')
    if UNIT_RANGE is not None:
        records = [u for u in records if u['unit'] in UNIT_RANGE]
    csvs[length] = records


def clean(name):
    return name.replace('-', '_').replace(' ', '_')


repeats = []
for length, records in tqdm(csvs.items(), desc='Lengths', total=len(csvs)):
    # Parse labels...check exact match...assuming you can sort
    units = []
    for unit in tqdm(records, desc='Parsing'):
        unit_f = F.parse(unit['label'])
        unit_expr = unit_f.to_expr(namer=clean)
        units.append(unit_expr)

    # This is O(n^2)
    unit_counts = Counter()
    for unit in tqdm(units, desc='Checking equivalence'):
        for comp_unit in unit_counts:
            if unit.equivalent(comp_unit):
                unit_counts[comp_unit] += 1
                break
        else:
            unit_counts[unit] += 1

    units_flat = unit_counts.most_common()
    units_flat = [(length, str(u), c) for (u, c) in units_flat]
    repeats.extend(units_flat)

repeats_df = pd.DataFrame(repeats, columns=['length', 'label', 'count'])
repeats_df.to_csv('repeats.csv')
