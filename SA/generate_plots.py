import pandas as pd
bench_data = pd.read_json('results.json')
print(bench_data['dummy10.txt'])