"""
Market Dataset-specific components

Aligned with the flat snapshot model:
  - EquitySnapshot: one row per equity quote capture
  - OptionSnapshot: one row per option quote capture

Properties match the upstream quote-snapshot vocabulary, which this project
mints itself and which resolves under MARKET_QUOTES. Note this is NOT the
feeds vocabulary (MARKET_FEEDS): the two model quotes differently and share
some local names, which is what made a single MARKET constant appear to
work while matching nothing.
"""