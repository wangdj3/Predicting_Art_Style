# Project 4 : Group Project
**Daniel Wang | Nicholas Nguyen | Steven Shields | Kaytlynn Skibo**

---
 ## The objective of this project is to use Convolutional Neural Networks to predict the style of a painting based on the image of a painting.
 
---
A note for anyone attempting to recreate the results of this notebook: please note that much of this code that was run in the notebook can be quite demanding. Some code may need to be broken up to run on your machine. Images and filepaths are not included, but can be created by running the code located in the notebook `02_Data_cleaning.ipynb`. Google colab was used as the primary enviornment this project's models were run in. Please keep this in mind when attempting to run as to clog your memory or lose your sanity as we did.

---


+ **Data** : `Wikiart_scraped.csv` , `raw_data.csv`, `clean_art.csv` | `df`,  `cleaned_df`
  + Source : [WikiArt | All Images]('https://www.kaggle.com/datasets/antoinegruson/-wikiart-all-images-120k-link')
  + Size : 124170 rows × 5 columns | 620,850 elements
  + Columns : `['Style', 'Artwork', 'Artist', 'Date', 'Link']`
  + Target : Classification on the `Style` column using the images downloaded through the links in the `Links` column
  
**Libraries Used**
+ Pandas
+ Numpy
+ Matplotlib
+ Scikit Learn
+ langid
+ requests
+ googletrans
+ copy
+ os 
+ io
+ sys
+ re
+ pathlib
+ tensorflow.keras
+ google.colab
+ seaborn
+ altair
+ plotly.express
+ nltk
  
| **Name**   | **Description** | **dtype** |
| ----------- | ----------- | ------- |
|`Style`|_The style of art the image has been annotated as_| object |
|`Artwork`|_The title of the artwork_|object|
|`Artist`|_The artist of the painting_|object|
|`Date`|_The Date the painting was created or assumed to be created_| int64|
|`Link`|_The link to images from the wikiart api_|object|
|`Language`|_The language of Artwork titles_|int64|
|`translated`|_The translated text of the artwork column_|object|
|`hex`|_The dominant color in the painting in hex_|object|
|`color`|_The dominant color in the painting_|object|
|`v_sent`|_The composite sentiment score of the artwork title_|float64|

**Model Performance on Data**
+ Baseline: 0.0397
+ `X` : images in the style directory in subfolders based on associated style
+ `y` : style of paintings
+ random_state = 2023
+ 51 classes
+ Parameters : 
  + Optimizer: `adam`
  + Filters: `5`
  + Kernel Size : `(7,7)`
  + Drop Outs: `.10 - .20`
  + BatchNormilization: Yes
  + Regularization: No
  + Rescaling : Yes
  + Epochs : 10
+ Training `acc/loss` : 0.1992 / 2.9073
+ Validation `acc/loss` : 0.1783 / 3.0341

---

**Primary Findings**
- Wikiart API has not been maintained well
- `image_dataset_from_directory` is incredibly useful and helpful
- Talk about finding in graph
- talk about finding in another graph
- Even if you have a large amount of data, too many classes can still badly affect the model

**Recommendations**
- Have some idea what you're stealing before you pull the heist
- Don't steal 10's of thousands of paintings at once- it really does just make everything more complicated
- Making a CNN machine learning model to predict is too hard.  Just kidnap an art expert next time
- Get a second opinion on what style some pieces are.  Even within the same genre, there are sometimes such a wide variation between works (to the untrained eye), that it’s hard to believe they really all are the same style.
- Don’t have so many different possible prediction classes. Definitely try to combine some of the similar categories.


**Conlusions**
+ Our model is a failed model. It predicts incorrectly 4/5ths of the time. That is not a case for a successful model. Our baseline was 0.0397 and although we achieved our metric of success, 0.178, it is nowhere near production ready and needs more time to develop. We've learn quite a bit from this experience and realize that predicting something as subjective as art is a difficult task, especially with 51 labels. I hope that you learn from our mistakes and find the same growth we have in this process.

**Next Steps**
- More time : Given another week to experiment with the following steps I strongly feel we could've gotten better results.
- More Processing power : this was a severe limitation on our project, as many models often had to run overnight on only 5 epochs.
- Combine like art styles (High/Early/Late Renaissance) : 51 classes is a tall order
- Higher resolution images : Could've possibly improved model accuracy, but would also increase amount of time needed to run
- Using only most populated classes : less classes and plenty of data to use, however this reduces the overall size of the data drastically
- Using models other than CNN : we put most of our focus into making this type of neural network work, but other types such as Deep Neural Networks or utilizing Transfer Learning could have improved our prediction ability

---

**In this Repository**
- `code` : this folder contains all notebooks associated with the project
  - `00_Problem_Statement` : In this notebook we will be discussing the problem we will be solving, why it is a problem, who this can help, and what our measures fo success will be. This will be the reoccuring topic and goal throughout these notebooks.
  - `01_Data_Collection` : Reading in the dataset we had found on Kaggle (sourced above) and checking it was read in correctly before dropping dead links and saving it to a separate csv.
  - `02_Data_Cleaning` : In this notebook we will be cleaning these data to prepare it for EDA and Modeling. We will check for null values, outliers, errors, and other attributes that would depreciate EDA and Modeling. We will engineer features to get further insight into these data. We will be using functions such as `to_date`, `lang_column`, and `cleaned_data`. The process here serves the purpose for preparing the dataframe for future notebooks.
  - `03_EDA` : 
  - `04_Preprocessing_Modeling` : In this notebook, we will be accessing the image directory of `Style` we had made in a previous notebook, `02_Data_Cleaning.ipynb`. We will set up the train and validation sets to train the model on and measure success by. Specifically, we will be using a CNN and testing various hyperparameters in efforts to create the best performing model.
  - `05_Conclusion_Recommendations` : > In this notebook we will give recommendations and come to a conclusion based upon the data collected in `01_Data_Collection`, analyzed in `03_EDA`, and modeled in `04_Preprocessing_Modeling` in this repository.
  - `streamlit_application` : Allows for uploads on images for model to be trained on. Predicts which styles correlate to the image with a confidence percentage.

- `data` : this folder contains all csvs collected or created throughout the project
  - `wikiart_scraped.csv`
  - `raw_data.csv`
  - `clean_art.csv`
  - `model.hdf5`


- `docs` : this folder contains documents related to the project 
  - `template`
  - `models_run`


- `images` : this folder contains all images downloaded from the dataset 
  - `data_visuals` : the visuals of the data made in `03_EDA` and `04_Preprocessing_Modeling`
  - `styles` : the unique instances of style found in the `clean_df` dataframe, the file path to these subfolders is made in `02_Data_Cleaning`

