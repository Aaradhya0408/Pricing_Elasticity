# E-commerce Pricing Optimisation Using Elasticity Analysis
HCLTech Career Shaper Programme - Project 44

## Files
- Pricing_Elasticity_Analysis.ipynb - main notebook (cleaning, EDA, elasticity models, simulation, A/B test, ethics checks)
- dataset.csv - raw (synthetic) sales data
- dataset_clean.csv - cleaned data produced by the notebook
- data_dictionary.csv - description of every column
- elasticity_results.csv - per-category elasticity and price recommendations
- generate_dataset.py - script that generated the synthetic dataset (seeded, reproducible)

## How to run
1. pip install pandas numpy scipy statsmodels matplotlib seaborn jupyter
2. Keep all files in the same folder
3. jupyter notebook Pricing_Elasticity_Analysis.ipynb, then Run All

## Notes
- Data is synthetic; unit costs in the profit simulation are assumptions.
- The A/B test is simulated from the fitted model to demonstrate the method.
