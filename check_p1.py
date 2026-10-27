import json
nb = json.load(open('notebooks/closed_vs_gd.ipynb', encoding='utf-8'))
for i,c in enumerate(nb['cells']):
    if c['cell_type']=='markdown':
        print('===MD', i, '===')
        src = ''.join(c['source'])
        print(src[:150])
        print()
