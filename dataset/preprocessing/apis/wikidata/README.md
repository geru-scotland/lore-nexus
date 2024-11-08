
# Wikidata Query Service
Thanks to the user Nicolas Raoul: 
https://opendata.stackexchange.com/users/754/nicolas-raoul♦
https://opendata.stackexchange.com/questions/201/database-of-fictional-characters

I have tinkered with the query and, thanks to ChatGPT:
```
SELECT ?itemLabel ?universeLabel WHERE {
  # Item's type is: fictional character, or sub-type, or sub-sub-type, etc
  ?item p:P31/ps:P31/wdt:P279* wd:Q95074.
  
  # Universo ficticio definido para el personaje
  ?item wdt:P1080 ?universe.

  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "en" .
  }
}
```