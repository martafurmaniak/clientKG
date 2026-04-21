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


# ─────────────────────────────────────────────────────────────────────────────
# Mock corroboration documents
# ─────────────────────────────────────────────────────────────────────────────
# Two small summarized documents used when RUN_MODE = "mock".
# Structure mirrors the real *_summarized.json format.

MOCK_CORROBORATION_DOCS = [
    {
        "pages": [
            {
                "page_number": 0,
                "offset": 0,
                "page_text": "Bank statement for ACC-001. Period: Jan-Mar 2024. Holder: John Smith.",
                "page_summary": (
                    "Bank statement for savings account ACC-001 held by John Smith at Alpine Bank. "
                    "Period January to March 2024. Opening balance CHF 260,000. "
                    "Debit of CHF 50,000 on 12 March 2024 to GlobalTech Ltd reference CONSULTING. "
                    "Credit of CHF 10,000 on 15 March 2024 from Robert Brown reference GIFT. "
                    "Closing balance CHF 220,000."
                ),
            },
            {
                "page_number": 1,
                "offset": 0,
                "page_text": "Account holder details and signatures.",
                "page_summary": (
                    "Account holder: John Smith, date of birth 1979-04-03, passport CH-9982211. "
                    "Joint holder: Sarah Smith, date of birth 1981-11-17. "
                    "Both holders provided wet signatures on 5 January 2020."
                ),
            },
        ],
        "document_summary": (
            "Alpine Bank statement for account ACC-001 (savings) covering Q1 2024. "
            "Account held by John Smith with Sarah Smith as joint holder. "
            "Two notable transactions: CHF 50,000 outgoing to GlobalTech Ltd "
            "and CHF 10,000 incoming from Robert Brown."
        ),
        "meta": {
            "summarized_at": "2024-04-01T10:00:00Z",
            "source_file": "alpine_bank_statement_q1_2024.pdf",
        },
    },
    {
        "pages": [
            {
                "page_number": 0,
                "offset": 0,
                "page_text": "Commercial register extract for TechVentures AG.",
                "page_summary": (
                    "Commercial register extract for TechVentures AG, CHE-123.456.789. "
                    "Registered address: Bahnhofstrasse 10, 8001 Zurich. "
                    "Founded 15 March 2015. Share capital CHF 100,000. "
                    "CEO and sole director: John Smith, born 1979-04-03. "
                    "The company develops enterprise software for financial institutions."
                ),
            },
        ],
        "document_summary": (
            "Official Swiss commercial register extract for TechVentures AG confirming "
            "John Smith as CEO and sole director, company registration number CHE-123.456.789, "
            "founded 2015 in Zurich."
        ),
        "meta": {
            "summarized_at": "2024-04-01T10:05:00Z",
            "source_file": "techventures_commercial_register.pdf",
        },
    },
]
