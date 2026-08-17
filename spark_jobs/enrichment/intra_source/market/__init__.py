"""
Market Dataset-specific components

Aligned with the flat snapshot model:
  - EquitySnapshot: one row per equity quote capture
  - OptionSnapshot: one row per option quote capture

Properties match the upstream quote-snapshot vocabulary, which this project
mints itself and which resolves under MARKET_QUOTES -- the only market
vocabulary. A second one (MARKET_FEEDS) once sat beside it with different names
for the same ideas; a single constant asked to name both is what made market
enrichment appear to work while matching nothing.
"""