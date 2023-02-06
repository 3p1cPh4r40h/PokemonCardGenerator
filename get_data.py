import pandas as pd
from pokemontcgsdk import Card

cards = Card.where(q='supertype:Pokémon')

cards_array = []
for card in cards:
    card_info = {
        "name": card.name,
        "subtypes": card.subtypes,
        "rules": card.rules,
        "types": card.types,
        "hp": card.hp,
        "attacks": card.attacks,
        "weaknesses": card.weaknesses,
        "convertedRetreatCost": card.convertedRetreatCost,
        "legalities": card.legalities,
        "evolvesFrom": card.evolvesFrom
    }
    cards_array.append(card_info)

card_df = pd.DataFrame(cards_array)
pd.to_pickle(card_df, 'card_df.pkl')
print(card_df.head())