"""Lane ``mcx``: MCX dated-contract daily history, session calendar, contract specs.

Outputs go to ``build/mcx/`` under the build root. See SPEC.md.
"""

LANE = "mcx"

#: Exchange symbol -> dataset asset key, for the four required products.
ASSET_BY_SYMBOL = {
    "GOLD": "gold",
    "SILVER": "silver",
    "COPPER": "copper",
    "CRUDEOIL": "crude",
}

MAIN_SYMBOLS = tuple(ASSET_BY_SYMBOL)
