import pandas as pd
import numpy as np

# ─── LOAD ───
# read data, check Data/doc.md for data description
pivoted = pd.read_csv('data_testing_storage/pivoted_stress_scores.csv')
# read CRSP data. 
crsp    = pd.read_csv('Data/crsp_daily.csv', dtype={'RET': str, 'vwretd': str})

# convert data types
crsp['date']   = pd.to_datetime(crsp['date'])
crsp['RET']    = pd.to_numeric(crsp['RET'],    errors='coerce')
crsp['vwretd'] = pd.to_numeric(crsp['vwretd'], errors='coerce')
pivoted['call_date'] = pd.to_datetime(pivoted['call_date'])

# calculate compound return for a given series of returns
'''
series = [0.01, -0.02, 0.03]  # Example returns
compound_ret = compound_return(series)
print(f"Compound Return: {compound_ret:.4f}")
'''
def compound_return(series):
    clean = series.dropna()
    if len(clean) == 0:
        return np.nan
    return (1 + clean).prod() - 1

# ─── COMPUTE CAR(0,1), VOL, MOM ───
results = []

for idx, row in pivoted.iterrows():
    symbol    = row['symbol']
    call_date = row['call_date']

    firm = crsp[crsp['TICKER'] == symbol].sort_values('date').reset_index(drop=True)

    if firm.empty:
        continue
    
    # Find day 0: the day after the call date (the next trading day)
    day0_candidates = firm[(firm['date'] > call_date) & (firm['date'] < call_date + pd.Timedelta(days=5))]  # Ensure we only look for the next trading day
    
    # check there are records for the firm
    if day0_candidates.empty:
        continue
    
    day0_idx = day0_candidates.index[0]
    
    # if there are no rows after day0, skip
    if day0_idx + 1 >= len(firm):
        continue
        
    day0     = firm.loc[day0_idx, 'date']
    day1 = firm.loc[day0_idx + 1, 'date']

    window_01 = firm.iloc[day0_idx:day0_idx + 2]
    
    # safety measure, not necessarily needed
    if len(window_01) < 2:
        continue

    # ── CAR(0,1) ─
    stock_01    = compound_return(window_01['RET'])
    market_01   = compound_return(window_01['vwretd'])
    car_01      = stock_01 - market_01

    results.append({
        'symbol':       symbol,
        'year':         row['year'],
        'quarter':      row['quarter'],
        'call_date':    call_date,
        'stress_pr':    row['stress_pr'],
        'stress_qa':    row['stress_qa'],
        'stress_whole': row['stress_whole'],
        'car_01':       car_01,
    })
    
    if idx % 500 == 0:
        print(f"Progress: {idx}/{len(pivoted)} | done: {len(results)}")

df_short = pd.DataFrame(results)
print(f"\nShort run dataset: {df_short.shape}")
print(f"Missing car_01: {df_short['car_01'].isna().sum()}")

df_short.to_csv('Data/new_new_new_CAR_1.csv', index=False)
print("✅ Saved new_new_CAR_1")


