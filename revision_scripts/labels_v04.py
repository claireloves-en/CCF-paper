"""
Feature labels that name what each encoded feature measures (Revision_Plan_V08.md, A4; review of CCF_MM_V03,
item 2). One map, used by the CCF_MM_V04 build (tables and text), the V03 figures and the feature dictionary,
so the paper never shows two names for one feature.

  category_3   is the quartile of the category's mean transaction amount, not the cost of an item
  category_2   necessity is a judgement assigned to the whole category, not a property of the purchase
  category_4*  likewise for purpose
  job_3        is O*NET Job Zone - 1: the preparation an occupation requires, not an individual's seniority
  job_4        is the occupation's O*NET 'Computers and Electronics' knowledge rating
  city_3       is the USDA Urban Influence Code condensed to metro / micropolitan / noncore
  city_*, loc_*, issue_transac_dist  describe the cardholder's county and home, not where the card was issued

RENAME maps the old display label (model_V26 cell 5's column_names_dict) to the new one.
"""
RENAME = {
    "Non-essential vs. essential": "Category necessity (coded)",
    "Average item cost": "Category typical amount (quartile)",
    "Purpose of purchase: household/wellbeing": "Category purpose (coded): household/wellbeing",
    "Purpose of purchase: entertainment/lifestyle": "Category purpose (coded): entertainment/lifestyle",
    "Purpose of purchase: shopping-related": "Category purpose (coded): shopping",
    "Interaction 1: Non-essential vs. essential*Average item cost": "Interaction 1: necessity × typical amount",
    "Interaction 2: Non-essential vs. essential*Purpose of purchase: household or wellbeing":
        "Interaction 2: necessity × household/wellbeing purpose",
    "City Rural-Urban": "County type (metro/micropolitan/noncore)",
    # the city attributes are county attributes of the cardholder's residence; nothing is about card issue
    "City socioeconomic status": "County income level (quartile)",
    "City social inequality level": "County income inequality (quintile)",
    "City high-speed fixed broadband coverage": "County fixed broadband coverage (quintile)",
    "City high-speed mobile broadband": "County 5G coverage (quintile)",
    "Location Cluster Card Issue": "Grid cell of cardholder's home",
    "Location Cluster Transaction": "Grid cell of merchant",
    "Distance between where card was issued and transaction made": "Distance from home to merchant (km)",
    "Occupation career level": "Occupation preparation (O*NET Job Zone)",
    "Occupation cyber knowledge": "Occupation computer knowledge",
}
# longest first, so a label that contains another is replaced whole
ORDERED = sorted(RENAME.items(), key=lambda kv: -len(kv[0]))


def relabel(s):
    """Replace every old label inside a string (exact labels and labels embedded in text)."""
    if not isinstance(s, str):
        return s
    for old, new in ORDERED:
        s = s.replace(old, new)
    return s
