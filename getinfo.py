from pygbif import occurrences as occ
from pygbif import registry
from pygbif import species
import pandas as pd


class Coral:
    def __init__(self):
        pass

    def get_species_name(self, name):
        """
        Get the species name from the taxon key
        """
        name = species.name_suggest(q = name, limit=1)
        df = pd.DataFrame(name)
        return df[['kingdom', 'phylum', 'class', 'order', 'family', 'scientificName', 'canonicalName']]




