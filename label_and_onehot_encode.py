import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np

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
card_df['attack_name_1'] = None
card_df['attack_name_2'] = None
card_df['attack_name_3'] = None
card_df['attack_name_4'] = None
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
            card_df.at[i, 'attack_text_{}'.format(j + 1)] = attack.name
            card_df.at[i, 'attack_text_{}'.format(j + 1)] = attack.text
            card_df.at[i, 'attack_damage_{}'.format(j + 1)] = attack.damage
            card_df.at[i, 'attack_convertedEnergyCost_{}'.format(j + 1)] = attack.convertedEnergyCost
    else:
        # set attack information to None or empty string
        for j in range(4):
            card_df.at[i, 'attack_text_{}'.format(j + 1)] = None
            card_df.at[i, 'attack_text_{}'.format(j + 1)] = None
            card_df.at[i, 'attack_damage_{}'.format(j + 1)] = None
            card_df.at[i, 'attack_convertedEnergyCost_{}'.format(j + 1)] = None

# Drop unnecesary attacks column
card_df = card_df.drop('attacks', axis=1)


card_df['weaknesses'] = [str(i) for i in card_df['weaknesses']]
card_df['weaknesses'] = card_df['weaknesses'].astype(str)

card_df['evolvesFrom'] = card_df['evolvesFrom'].astype(str)

# Store the label encoders used
label_encoders = {}
attack_damage_encoder = LabelEncoder()
attack_convertedEnergyCost_encoder = LabelEncoder()
# Store label encoded data
le_card_df = pd.DataFrame()
print(card_df.columns)
# Label encode data
for column in ['name', 'types', 'hp', 'weaknesses', 'convertedRetreatCost', 'evolvesFrom']:
    label_encoder = LabelEncoder()
    le_card_df[column] = label_encoder.fit_transform(card_df[column].astype(str))
    label_encoders[column] = label_encoder

# Label encode attacks and store them in the label encoded card dictionary
le_card_df['attack_damage'] = attack_damage_encoder.fit_transform(card_df['attack_damage_1'])
le_card_df['attack_convertedEnergyCost'] = attack_convertedEnergyCost_encoder.fit_transform(card_df['attack_convertedEnergyCost_1'])

# Create a dataframe to store the label decoding dictionaries
label_decoding_df = pd.DataFrame(columns=['column_name', 'encoding', 'decoding'])

# Loop through each column's label encoder and store the label decoding dictionary
for column, label_encoder in label_encoders.items():
    decoding = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    temp_df = pd.DataFrame({'column_name': [column] * len(decoding), 'encoding': list(decoding.keys()), 'decoding': list(decoding.values())})
    label_decoding_df = pd.concat([label_decoding_df, temp_df], ignore_index=True)





# Store the onehot encoders used
onehot_encoders = {}
# Store onehot encoded data
oh_card_dict = {}
# Create an initial decoding dictionary
oh_decoding_dict = {}
# Onehot encode data
for column in ['subtypes', 'rules', 'legalities']:
    onehot_encoder = OneHotEncoder()
    encoded_data = onehot_encoder.fit_transform(card_df[column].astype(str).values.reshape(-1, 1))  
    oh_card_dict[column] = encoded_data.toarray()
    onehot_encoders[column] = onehot_encoder
    # Store the inverse transformation
    oh_decoding_dict[column] = {}
    for i, category in enumerate(onehot_encoder.categories_[0]):
        oh_decoding_dict[column][i] = category

# Store the onehot encoders used for attacks
attack_name_encoder = OneHotEncoder()
attack_text_encoder = OneHotEncoder()
onehot_encoders['attack_name'] = attack_name_encoder
oh_decoding_dict['attack_name'] = {}
for i, category in enumerate(onehot_encoder.categories_[0]):
    oh_decoding_dict['attack_name'][i] = category

onehot_encoders['attack_text'] = attack_text_encoder
oh_decoding_dict['attack_text'] = {}
for i, category in enumerate(onehot_encoder.categories_[0]):
    oh_decoding_dict['attack_text'][i] = category


card_df.fillna('NA')
# Prepare name and text for label one hot encoding
first_attack_names = card_df['attack_name_1']
first_attack_texts = card_df['attack_text_1']

names_values = first_attack_names.values.reshape(-1,1)
texts_values = first_attack_texts.values.reshape(-1,1)

# Onehot encode attacks
attack_encoded_name_data = attack_name_encoder.fit_transform(names_values)
attack_encoded_text_data = attack_text_encoder.fit_transform(texts_values)

# Store encoded data in dictionary
oh_card_dict['attack_name'] = attack_encoded_name_data.toarray()
oh_card_dict['attack_text'] = attack_encoded_text_data.toarray()

# Convert arrays to lists
oh_card_dict = {key: value.tolist() for key, value in oh_card_dict.items()}

# Concatenate encoded data into a single dataframe
oh_card_df = pd.DataFrame.from_dict(oh_card_dict)

# Create decoding dictionary by inverting the encoding using 
# inverse_transform method of each OneHotEncoder object
oh_decoding_dict = {}

for column, onehot_encoder in onehot_encoders.items():
    oh_decoding_dict[column] = {}
    for i, category in enumerate(onehot_encoder.categories_[0]):
        oh_decoding_dict[column][i] = category

pd.to_pickle(le_card_df, 'test_data\label_encoded_df.pkl')
pd.to_pickle(oh_card_df, 'test_data\oh_encoded_df.pkl')
pd.to_pickle(label_decoding_df, 'test_data\label_decoding_dict.pkl')
pd.to_pickle(oh_decoding_dict, 'test_data\oh_decoding_dict.pkl')
print(le_card_df.head())
print(oh_card_df.head())
#print(label_decoding_df)
#print(oh_decoding_dict)

# Decode the label encoded data
for column, label_encoder in label_encoders.items():
    le_card_df[column] = le_card_df[column].map(dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_)))

# Decode the onehot encoded data
for column in ['subtypes', 'rules', 'legalities', 'attack_name', 'attack_text']:
    decoding_dict = oh_decoding_dict[column]
    decoded_data = []
    for i in range(len(oh_card_df)):
        row = oh_card_df.iloc[i][column]
        decoded_values = [decoding_dict[j] for j, val in enumerate(row) if val == 1]
        decoded_data.append(', '.join(decoded_values))
    le_card_df[column] = decoded_data

# Combine the label encoded and onehot encoded data into a single dataframe
decoded_card_df = le_card_df.join(card_df[['types', 'hp', 'weaknesses', 'convertedRetreatCost', 'evolvesFrom', 'attack_damage_1', 'attack_convertedEnergyCost_1']])

# Rename columns
decoded_card_df = decoded_card_df.rename(columns={'attack_damage_1': 'attack_damage', 'attack_convertedEnergyCost_1': 'attack_convertedEnergyCost'})

# Reorder columns
decoded_card_df = decoded_card_df[['name', 'subtypes', 'rules', 'types', 'hp', 'weaknesses', 'convertedRetreatCost', 'legalities', 'evolvesFrom', 'attack_name', 'attack_text', 'attack_damage', 'attack_convertedEnergyCost']]

# Display the decoded data
print(decoded_card_df.head())
