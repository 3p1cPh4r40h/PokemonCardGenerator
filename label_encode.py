import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Import card dataframe
card_df = pd.read_pickle("data\card_df.pkl")

# Verify length of card dataframe
print(len(card_df))

print(card_df.head())


# Prepare values for encoding
card_df['hp'] = card_df['hp'].astype(int)
card_df['convertedRetreatCost'] = card_df['convertedRetreatCost'].fillna(0)
card_df['convertedRetreatCost'] = card_df['convertedRetreatCost'].astype(int)

card_df['name'] = card_df['name'].astype(str)

card_df['subtypes'] = [str(i) for i in card_df['subtypes']]
card_df['subtypes'] = card_df['subtypes'].astype(str)

card_df['rules'] = card_df['rules'].astype(str)

card_df['types'] = card_df['types'].astype(str)

# create new columns for each element of an attack
card_df['attack_name'] = None

card_df['attack_text'] = None

card_df['attack_damage'] = None

card_df['attack_convertedEnergyCost'] = None


for i, row in card_df.iterrows():
    attacks = row['attacks']
    if attacks:
        # record first attack
       
        attack = attacks[0]
        card_df.at[i, 'attack_name'] = attack.name
        card_df.at[i, 'attack_text'] = attack.text
        card_df.at[i, 'attack_damage'] = attack.damage
        card_df.at[i, 'attack_convertedEnergyCost'] = attack.convertedEnergyCost
    else:
        # set attack information to None or empty string
        
        card_df.at[i, 'attack_name'] = None
        card_df.at[i, 'attack_text'] = None
        card_df.at[i, 'attack_damage']= None
        card_df.at[i, 'attack_convertedEnergyCost'] = None

# Drop unnecesary attacks column
card_df = card_df.drop('attacks', axis=1)


card_df['weaknesses'] = [str(i) for i in card_df['weaknesses']]
card_df['weaknesses'] = card_df['weaknesses'].astype(str)

card_df['evolvesFrom'] = card_df['evolvesFrom'].astype(str)


# Store the label encoders used to encode each column
encoders = {}
le_card_df = pd.DataFrame()

for column in card_df.columns:
    if card_df[column].dtype == 'object':
        label_encoder = LabelEncoder()
        le_card_df[column] = label_encoder.fit_transform(card_df[column].astype(str))
        encoders[column] = label_encoder

# Create a dataframe to store the decoding dictionaries
decoding_dict = pd.DataFrame(columns=['column_name', 'encoding', 'decoding'])

# Loop through each column's label encoder and store the decoding dictionary
for column, label_encoder in encoders.items():
    decoding = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    temp_df = pd.DataFrame({'column_name': [column] * len(decoding), 'encoding': list(decoding.keys()), 'decoding': list(decoding.values())})
    decoding_dict = pd.concat([decoding_dict, temp_df], ignore_index=True)

pd.to_pickle(le_card_df, 'data\label_encoded_df.pkl')
pd.to_pickle(decoding_dict, 'data\label_decoding_dict.pkl')
print(le_card_df.head())
print(decoding_dict)
