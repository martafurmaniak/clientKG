"""
Mock input data: a short client history document + entity/relationship ontologies.
Kept intentionally tiny so qwen2.5:7b can handle it on a laptop.
"""

DOCUMENT_TEXT = """
Client Profile - John Smith

John Smith is a 45-year-old private banking client at Alpine Bank.
He is the CEO of TechVentures AG, a Swiss software company founded in 2015.
John's wife, Sarah Smith, is also registered as a joint account holder.

John holds a savings account (ACC-001) with a current balance of CHF 250,000 and
a brokerage account (ACC-002) holding 500 shares of TechVentures AG stock valued at CHF 150,000.

On 12 March 2024, John transferred CHF 50,000 from ACC-001 to an external account
owned by GlobalTech Ltd, citing a consulting payment.
On 15 March 2024, Sarah Smith received CHF 10,000 into ACC-001 described as a gift from her father, Robert Brown.

TechVentures AG maintains a corporate account (ACC-003) at Alpine Bank with a balance of CHF 500,000.
John Smith is listed as the beneficial owner of TechVentures AG.
"""

# Pages for the relationship extraction agent (it processes one page at a time)
DOCUMENT_PAGES = [DOCUMENT_TEXT]

ENTITY_ONTOLOGY = {
    "people": {
        "description": "A natural person",
        "attributes": ["name", "age", "role", "relationship_to_client"]
    },
    "organisations": {
        "description": "A legal entity or company",
        "attributes": ["name", "type", "founded_year", "industry"]
    },
    "assets": {
        "description": "A financial asset or account",
        "attributes": ["asset_id", "asset_type", "value_chf", "owner"]
    },
    "transactions": {
        "description": "A financial transaction",
        "attributes": ["transaction_id", "date", "amount_chf", "from_account", "to_party", "description"]
    }
}

RELATIONSHIP_ONTOLOGY = {
    "OWNS": {
        "description": "A person or org owns an asset",
        "from": ["people", "organisations"],
        "to": ["assets"]
    },
    "EMPLOYED_BY": {
        "description": "A person is employed by an organisation",
        "from": ["people"],
        "to": ["organisations"]
    },
    "LEADS": {
        "description": "A person leads/is CEO of an organisation",
        "from": ["people"],
        "to": ["organisations"]
    },
    "RELATED_TO": {
        "description": "Two people are family members",
        "from": ["people"],
        "to": ["people"]
    },
    "INVOLVED_IN": {
        "description": "A person or org is involved in a transaction",
        "from": ["people", "organisations"],
        "to": ["transactions"]
    },
    "BENEFICIAL_OWNER_OF": {
        "description": "A person is the beneficial owner of an organisation",
        "from": ["people"],
        "to": ["organisations"]
    },
    "ACCOUNT_AT": {
        "description": "An asset (account) is held at an organisation (bank)",
        "from": ["assets"],
        "to": ["organisations"]
    }
}
