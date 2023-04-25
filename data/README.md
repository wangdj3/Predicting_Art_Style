# data Directory
> This directory contains CSVs collected or modified in this repository, it also contains `model_checkpoints` which contains the best iteration of all models run an pickles in which the models were saved for our streamlit app.
---

**CSVs**
- `Wikiart_scraped.csv` is the data collected from the kaggle competition and was directly downloaded from the site.
- `raw_data.csv` is the `Wikiart_scraped.csv` with dead links to prevent those links from reaching the image scraper.
- `clean_art.csv`is the data after it had been cleaned and contains engineered features covered in the `02_Data_cleaning.ipynb` notebook