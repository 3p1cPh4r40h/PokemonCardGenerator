#%%
import pandas as pd
import matplotlib.pyplot as plt
card_df = pd.read_pickle('data/card_df.pkl')
sample = card_df.sample(100)
#print(sample)
#sample.to_clipboard()
#print(sample.isnull().sum())
#sample.isnull().sum().to_clipboard()
#dupcheck = .astype(str).duplicated()
#print(dupcheck.sum())
#card_df.name.value_counts().nlargest(10).plot(kind='bar',xlabel='Name',ylabel='Count',title='Most Common Names')
#sample.name.value_counts().plot(kind='bar',xlabel='Name',ylabel='Count',title='Amount of Each Type')
print(sample.groupby('types'))
#sample.astype(str).loc[].name.value_counts().nlargest(10).plot(kind='bar',xlabel='Name',ylabel='Count',title='Amount of Each Type')
# %%
