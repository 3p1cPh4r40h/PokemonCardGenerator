import pandas as pd
import matplotlib.pyplot as plt
card_df = pd.read_pickle('card_df.pkl')
sample = card_df.sample(1000)
#print(sample)
#sample.to_clipboard()
#print(sample.isnull().sum())
#sample.isnull().sum().to_clipboard()
#dupcheck = .astype(str).duplicated()
#print(dupcheck.sum())
sample.types.value_counts().plot(kind='bar')