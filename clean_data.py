import pandas as pd
from sklearn.preprocessing import LabelEncoder

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

# Store the label encoders used to encode each column
encoders = {}
ohe_card_df = pd.DataFrame()

for column in card_df.columns:
    if card_df[column].dtype == 'object':
        label_encoder = LabelEncoder()
        ohe_card_df[column] = label_encoder.fit_transform(card_df[column].astype(str))
        encoders[column] = label_encoder

# Create a dataframe to store the decoding dictionaries
decoding_dict = pd.DataFrame(columns=['column_name', 'encoding', 'decoding'])

# Loop through each column's label encoder and store the decoding dictionary
for column, label_encoder in encoders.items():
    decoding = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    temp_df = pd.DataFrame({'column_name': [column] * len(decoding), 'encoding': list(decoding.keys()), 'decoding': list(decoding.values())})
    decoding_dict = pd.concat([decoding_dict, temp_df], ignore_index=True)

pd.to_pickle(ohe_card_df, 'ohe_card_df.pkl')
pd.to_pickle(decoding_dict, 'decoding_dict.pkl')
print(ohe_card_df.head())
print(decoding_dict)
