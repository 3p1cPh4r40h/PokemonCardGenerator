import pandas as pd
from sklearn.preprocessing import LabelEncoder

# label encoder object used to label encode data for one hot encoding
label_encoder = LabelEncoder()

# Import card dataframe
card_df = pd.read_pickle("card_df.pkl")

# Verify length of card dataframe
print(len(card_df))

print(card_df.head())


# Prepare values for one-hot-encoding
card_df['hp'] = card_df['hp'].astype(int)
card_df['convertedRetreatCost'] = card_df['convertedRetreatCost'].fillna(0)
card_df['convertedRetreatCost'] = card_df['convertedRetreatCost'].astype(int)

card_df['name'] = card_df['name'].astype(str)

card_df['subtypes'] = [str(i) for i in card_df['subtypes']]
card_df['subtypes'] = card_df['subtypes'].astype(str)

card_df['rules'] = card_df['rules'].astype(str)

card_df['types'] = card_df['types'].astype(str)

# create new columns for each element of an attack
card_df['attack_text_1'] = None
card_df['attack_text_2'] = None
card_df['attack_text_3'] = None
card_df['attack_text_4'] = None
card_df['attack_damage_1'] = None
card_df['attack_damage_2'] = None
card_df['attack_damage_3'] = None
card_df['attack_damage_4'] = None
card_df['attack_convertedEnergyCost_1'] = None
card_df['attack_convertedEnergyCost_2'] = None
card_df['attack_convertedEnergyCost_3'] = None
card_df['attack_convertedEnergyCost_4'] = None

for i, row in card_df.iterrows():
    attacks = row['attacks']
    if attacks:
        # loop over each attack
        for j in range(len(attacks)):
            attack = attacks[j]
            card_df.at[i, 'attack_text_{}'.format(j + 1)] = attack.text
            card_df.at[i, 'attack_damage_{}'.format(j + 1)] = attack.damage
            card_df.at[i, 'attack_convertedEnergyCost_{}'.format(j + 1)] = attack.convertedEnergyCost
    else:
        # set attack information to None or empty string
        for j in range(4):
            card_df.at[i, 'attack_text_{}'.format(j + 1)] = None
            card_df.at[i, 'attack_damage_{}'.format(j + 1)] = None
            card_df.at[i, 'attack_convertedEnergyCost_{}'.format(j + 1)] = None

card_df.drop('attacks', axis=1)

card_df['weaknesses'] = [str(i) for i in card_df['weaknesses']]
card_df['weaknesses'] = card_df['weaknesses'].astype(str)

card_df['evolvesFrom'] = card_df['evolvesFrom'].astype(str)


for column in card_df.columns:
    if card_df[column].dtype == 'object':
        card_df[column] = label_encoder.fit_transform(card_df[column].astype(str))

# Print columns of dataframe to verify conversion
columns = card_df.columns
for column in columns:
    print(card_df[column])


ohe_card_df = pd.get_dummies(card_df)
print(ohe_card_df)
pd.to_pickle(ohe_card_df, 'ohe_card_df.pkl')