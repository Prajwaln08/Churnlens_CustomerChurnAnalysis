"""Customer Churn Analytics Suite - production modules.

Every function here was first prototyped in one of the numbered notebooks in
Code/ and then promoted. Each module's docstring names its origin notebook.

Modules
-------
config       paths, seeds and settings read from the project .env
data_loader  loaders for the three datasets (Telco, Online Retail II, MegaFon)
eda          reusable, palette-safe plotting helpers
features     feature engineering per lens + train/test splitting
models       model training and artifact persistence for all five lenses
evaluation   metrics, lift/gains, qini, cluster profiles, revenue-at-risk
pipeline     schema contracts, validation, lens detection, scoring, CLI
"""
