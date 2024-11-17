"""
*****************************************************
 * Universidad del País Vasco (UPV/EHU)
 * Facultad de Informática - Donostia-San Sebastián
 * Asignatura: Procesamiento de Lenguaje Natural
 * Proyecto: Lore Nexus
 *
 * File: querys.py
 * Author: geru-scotland
 * GitHub: https://github.com/geru-scotland
 * Description:
 ***
 """
from SPARQLWrapper import SPARQLWrapper, JSON
import time

sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
sparql.setReturnFormat(JSON)

occupations = {
    "scientist": "Q82955",
    "artist": "Q82960",
    "physicist": "Q169470",
    "biologist": "Q864503",
    "chemist": "Q593644",
    "mathematician": "Q170790",
    "philosopher": "Q4964182",
    "historian": "Q1930187",
    "painter": "Q1028181",
    "composer": "Q36834",
    "computer_scientist": "Q82594",
    "writer": "Q36180"
}

start_year = 1400
end_year = 2000
results = []

for occupation, occupation_id in occupations.items():
    try:
        query = f"""
        SELECT ?person ?personLabel WHERE {{
          ?person wdt:P31 wd:Q5;                # humano
                  wdt:P106 wd:{occupation_id}; 
                  wdt:P569 ?birth;             
                  wikibase:sitelinks ?sitelinks. # me baseo en numero de links para determina cuan relevante es

          FILTER (?sitelinks > 30)
          FILTER (YEAR(?birth) >= {start_year} && YEAR(?birth) <= {end_year})

          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        ORDER BY DESC(?sitelinks)
        LIMIT 5000
        """

        sparql.setQuery(query)
        response = sparql.query().convert()

        results.extend(response['results']['bindings'])

        time.sleep(1)

    except Exception as e:
        print(f"Error with {occupation}: {e}")

with open("raw_data/extremely_relevant_figures2.txt", "w", encoding="utf-8") as file:
    for result in results:
        name = result['personLabel']['value']
        file.write(name + "\n")

print("Query finished and data stored in raw_data/extremely_relevant_figures2.txt")
