# Portfolio Projects

This repository contains my Portfolio Projects on various data analysis and data science topics — from python practice and data visualisation to machine learning and natural language processing.

### 34 Supervised Machine Learning (Time-Series Data Prediction) [Electricity Consumption Forecast (SE)](./swedish_companies_real_cases/Energy_prediction/energy_prediction.ipynb)

- `Objective`: Forecast daily electricity consumption in Sweden for 2022.
- `Potential Business Impact`: more accurate budgeting and potential cost savings. 
- `Final Model R^2 Score`: `0.96`
- `Final Model MAE`: `9.94`
- `Dataset`: ~2K rows × 8 columns.
- `Assumption`: Accurate forecast for the following day is available.
- `Features`: Temperature data (Stockholm, Gothenburg, Lund), Holidays, GWh lags for 7 days.
- `Source`: Vattenfall - Swedish multinational power company. 

[Here](./swedish_companies_real_cases/Energy_prediction/Electricity%20Consumption%20Forecast%20(SE).pdf) you can find a short project presentation. 

### 33 Deep Learning (Image Recognition) ['Deep Transfer Learning for Land Cover Classification'](./educational_case_studies/ml/deep_learning/land_image_classification/land_usage_classifier.ipynb)

- `Objective`: develop land cover classifier trained on satellite images.
- `Method`: transfer learning with Wide Residual Networks (`50` CNNs) on the Land Cover (EuroSAT) dataset.
- `Results`: `95%` accuracy with `0.03` loss on test data.
- `Optimizations`: boost efficiency with gradient clipping, adaptive learning rates, and data augmentation.
- `Dataset`: `27,000` labeled images from Sentinel-2 satellite, data obtained via copernicus.eu (EU project).
- `Applications`: maps correction, urban planning, natural catastrophe prediction, precision agriculture.
- `Resourses`: ["Deep Transfer Learning for Land Use and Land Cover Classification: A Comparative Study"](https://www.mdpi.com/1424-8220/21/23/8083)

[Here](./educational_case_studies/ml/deep_learning/land_image_classification/Deep%20Transfer%20Learning%20for%20Land%20Cover%20Classification.pdf) you can find a short project presentation. 


-----------------------

### 32 EDA (Time-Series Data, Data Scraping) ['EU Energy Generation vs Consumption'](./educational_case_studies/statistics_and_visualisation/EU_energy/eda_EU_energy_handl_comp.ipynb)

Comparative Analysis of Energy Generation and Consumption done by **Sweden**, **Germany** and **Poland** during 8 recent years.

The goal of this project is to analyse the energy generation and consumption done by EU nations which have different levels of economic development, sociopolitical structure and influence of religion within the country. Specifically, `we’ll look at data flows to understand whether there are patterns charactarising eacREh nation’s energy handling strategy`. We are going to use free public datasets dedicated to energy handling within Europe, scraped from https://transparency.entsoe.eu. 

Short project [**presentation**](./educational_case_studies/statistics_and_visualisation/EU_energy/Comparing%20Energy%20Generation%20and%20Consumption%20in%20EU%20Nations.pdf).

-----------------------

### 31. Supervised Machine Learning (Gradient Boosting on Decision Trees): ['Predicting Extreme Weather Events using Gradient Boost'](/wids_datathon_2023/catboost.ipynb)

We've done some initial investigation and visualization withing EDA previously and came to the conclusion that the dataset is fairy complicated and there are no obvious clues on how to manually select features without proper domain knowledge. One obvious issue for potential feature selection that we've been able to find is increase of temperatures in the testing set, compared to training set due to climate change. Considering that this issue can cause data drift we are going to try to mitigate this issue using Adversarial Validation. For prediction itself we are going to try using gradient boosting model CatBoost, which is quite resilient to outliers and overfitting. For hyperparameter optimisation we are going to use Bayesian Optimizer.

-----------------------


### 30. EDA (Time-Series Data): ['Exploratory Analysis of Climate Data'](./wids_datathon_2023/eda_temp_prediction.ipynb)

Exploratory Data Analysis (EDA for short) refers to the process of discovering patterns and relationships within a dataset. It is about making sense of the data at hand and getting to know it before modeling or analysis. It is an important step prior to model building and can help us formulate further questions and areas for investigation. Within this project we are going work with the `training_data.csv` and `test_data.csv` provided by partners of [WiDS Datathon 2023](https://www.kaggle.com/competitions/widsdatathon2023/data). 

-----------------------


### 29. Deep Learning (RNNs & LSTMs): ['Temperature Forecast'](/ml/deep_learning/time_series/temp_prediction.ipynb)

The goal of this project is to compare performance of two deep learning algorithms on time-series data: Tensorflow's stacked layers of Simple Recurrent Neural Network (RNN) and Long-Short-Term Memory Networks (LSTMs), using MAE and RMSE as evaluation metrics.

-----------------------

### 28. Unsupervised Machine Learning (K-Means++): ["Masculinity Survey Insights"](/ml/unsupervised_learning/k_means/clustering_masculinity_survey.ipynb)

In this project, we will be investigating the way people think about masculinity by applying the **K-Means++** algorithm to data from  <a href="https://fivethirtyeight.com/" target = "_blank">FiveThirtyEight</a>. FiveThirtyEight is a popular website known for their use of statistical analysis in many of their stories.

-----------------------

### 27. Supervised Machine Learning (Ensemble Technique): ["OKCupid Dating App"](/ml/supervised_learning/ensemble/dating_app_profiles_analysis.ipynb)

The goal of this project is to find out whether it's possible to predict person's age and religious views via machine learning models, with sufficient accuracy, based on dating app profiles' data from OKCupid.  We are going to evaluate accuracy of different supervised models separately as well as create ensembles and assess resulting accuracy's shifts. Dataset is provided by Codecademy.com.

-----------------------

### 26. NLP: ["Movie Reviews"](./nlp/sent_analysis/nlp_portfolio_project.ipynb)

The goal of this project is to get insight into IMDB movie reviews on different layers using various NLP techniques and libraries. A dataset that we are going to use is meant mainly for **binary sentiment classification** and performing this particular type of analysis will, indeed, be our primary goal, but additionally, we'll **use POS tagging and chunking to find commonly used noun and verb phrases** and **find the most common movie topics users wrote about (topic modelling using LDA)**. 

----------------------

### 25. SQL (Windows Functions): ["Climate Change"](./sql/windows_functions/climate_change.ipynb)

The goal of this project is to practice SQL windows functions, ascertaining different climate change insights within "Global Land and Ocean Temperatures" dataset in the process.

----------------------

### 24. Unsupervised Learning (K-Means++): ["Handwriting Recognition"](/ml/unsupervised_learning/k_means/k_means_handwriting_rec.ipynb)

In this project, we will be using **K-Means++** clustering algorithm in scikit-learn inplementation on sklearn digits dataset to cluster images of handwritten digits.

----------------------

### 23. Supervised Machine Learning (Ensemble Technique - Random Forest): ["Predicting Income"](./ml/supervised_learning/random_forest/forest_income_project.ipynb)

In this project, we will be using an ensemble machine learning technique - **Random Forest**. We are going to compare its performance with **Dicision Tree** algorithm, on a dataset containing [census information from UCI’s Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Adult/). By using census data with a Random Forest and Dicision Tree, we will try to predict whether a person makes more than $50,000.

----------------------

### 22. Supervised Machine Learning (Decision Trees): ["Predict Continent and Language"](./ml/supervised_learning/trees/trees_find_flag.ipynb)

In this project, we’ll use **Decision Trees** to predict the continent a country is located on, and its language based on features of its flag and other country's properties. For instance, some colors are good indicators as well as the presence or absence of certain shapes could give one a hint. We’ll explore which features are the best to use and will create several Decision Trees to compare their results. The [**Flags Data Set**](https://archive.ics.uci.edu/ml/datasets/Flags) used in this project is provided by UCI’s Machine Learning Repository.

----------------------

### 21. Supervised Machine Learning (Support Vector Machines): ["Predict Baseball Strike Zones"](/ml/supervised_learning/svm_models/baseball_strike_zones.ipynb)

In this project, we will use a **Support Vector Machines**, trained using a [`pybaseball`](https://github.com/jldbc/pybaseball) dataset, to find strike zones decision boundary of players with different physical parameters.

----------------------

### 20. Supervised Machine Learning (Naive Bayes): ["Newsgroups Similarity"](./ml/supervised_learning/naive_bayes/newsgroups_simularity.ipynb)

In this project we will apply Scikit-learn’s **Multinomial Naive Bayes Classifier** to Scikit-learn’s example datasets to find which category combinations are harder for it to distinguish. We are going to achieve that by reporting the accuracy of several variations of the classifier that were fit on different categories of newsgroups.

----------------------

### 19. Supervised Machine Learning (Logistic Regression): ["Chances of Survival on Titanic"](/ml/supervised_learning/logistic_regression/logistic_regression_titanic.ipynb)

The goal of this project is to create a **Logistic Regression Model** that predicts which passengers survived the sinking of the Titanic, based on features like age, class and other relevant parameters.

-----------------------

### 18. Supervised Machine Learning (K-Nearest Neighbor classifier and others): ["Twitter Classification"](./ml/supervised_learning/k_nearest_neighbours/twitter_classification.ipynb)

The first goal of this project is to predict whether a tweet will go viral using a **K-Nearest Neighbour Classifier**. The second is to determine whether a tweet was sent from New York, London, or Paris using Logistic Regression and Naive Bayes classifiers with vectorization of different sparsity levels applied to their features. To even out gaps between number of tweets in different cities we'll apply text augmentation by BERT.

-----------------------

### 17. Supervised Machine Learning (Multiple Linear Regression): ["Tendencies of Housing Prices in NYC"](./ml/supervised_learning/linear_regression/multiple_linear_regression/prices_tendencies_for_housing.ipynb)

The goal of this project is to get an insight to the range of factors that influence NYC apartments' price formation, predict prices for several random apartments based on these insights with the help of **Multiple Linear Regression Model** and visualise some results using 2D and 3D graphs.

-----------------------

### 16. Supervised Machine Learning (Simple Linear Regression): ["Honey Production"](./ml/supervised_learning/linear_regression/simple_linear_regression/honey_production.ipynb)

The goal of this project is predicting honey production during upcoming years (till 2050) using **Simple Linear Regression Model** and some visualizations.

-----------------------

### 15. NLP, Feature Modeling (tf-idf): ["News Analysis"](./nlp/language_quantification/news_analysis/language_quantification.ipynb)

In this project we will use **"term frequency-inverse document frequency"** (tf-idf) to analyze each article’s content and uncover the terms that best describe each article, providing quick insight into each article’s topic.

-----------------------

### 14. NLP, Feature Modeling (Word Embeddings): ["U.S.A. Presidential Vocabulary"](./nlp/language_quantification/presidential_vocabulary/us_presidential_vocabulary.ipynb)

The goal of this project is to analyze the inaugural addresses of the presidents of the United States of America using **word embeddings**. By training sets of word embeddings on subsets of inaugural addresses, we can learn about the different ways in which the presidents use language to convey their agenda.

-----------------------

### 13. NLP, Topics Analysis (Chunking): ["Discover Insights into Classic Texts"](./nlp/language_parsing/language_parsing.ipynb)

The goal of this project is to discover the main themes and some other details from the two classic novels: Oscar Wilde’s **"The Picture of Dorian Gray"** and Homer’s **"The Iliad"**.  To achieve it, we are going to use `nltk` methods for preprocessing and creating Tree Data Structures, after which, we will apply filters to those Structures to get some desired insights.

-----------------------

### 12. NLP (Sentiment Analysis), Supervised Machine Learning (Ensemble Technique): ["Voting System"](./nlp/voting_system/nltk_scikitlearn_combined.ipynb)

The goal of this project is to create a voting system for *bivariant sentiment analysis* of any type of short reviews. To achieve this we are going to combine **Naive Bayes** algorithm from `nltk` and similar algorithms from `scikit-learn`. This combination should increase the accuracy and reliability of the confidence percentages.

-----------------------

### 11. NLP, Sentiment Analysis (Naive Bayes Classifier): ["Simple Naive Bayes Classifier"](./nlp/naive_bayes_classifier/naive_bayes_classifier.ipynb)

The goal of this project is to build a simple **Naive Bayes Classifier** using `nltk toolkit`, and after that: train and test it on Movie Reviews corpora from `nltk.corpus`.

-----------------------

### 10. Analysis via SQL: ["Gaming Trends on Twitch"](./sql/twitch_data_extraction_and_visualization/twitch_data_extraction_and_visualisation.ipynb)

The goal of this project is to analyse gaming trends with SQL and visualise them with Matplotlib and Seaborn. 

-----------------------

 ### 9. Statistical Analysis and Visualisation: ["Airline Analysis"](/statistics_and_visualisation/airline_analysis/airline_analysis.ipynb)

The goal of this project is to guide airline company clients' decisions by providing summary statistics that focus on one, two or several variables and visualise its results. 

-----------------------

### 8. Statistical Analysis and Visualisation: ["NBA Trends"](./statistics_and_visualisation/associations_nba/nba_trends.ipynb)

In this project, we’ll analyze and visualise data from the NBA (National Basketball Association) and explore possible associations.

-----------------------

### 7. Statistical Analysis and Visualisation: ["Titanic"](./statistics_and_visualisation/quant_and_categorical_var/titanic.ipynb)

The goal of this project is to investigate whether there are some correlations between the different aspects of physical and financial parameters and the survival rates of the Titanic passengers.

-----------------------

### 6. Data Transformation: ["Census Data Types Transformation"](./statistics_and_visualisation/census_datatypes_transform/census_datatypes_transform.ipynb)

The goal of this project is to use pandas to clean, organise and prepare recently collected census data for further usage by machine learning algorithms.

-----------------------

### 5. Data Transformation: ["Company Transformation"](./statistics_and_visualisation/data_transformation/company_transformation.ipynb)

The goal of this project is to apply data transformation techniques to better understand the company’s data and help to answer important questions from the management team.


-----------------------

### 4. Statistical Analysis and Visualisation: ["Roller Coasters"](./statistics_and_visualisation/roller_coasters/roller_coaster.ipynb)
The goal of this project is visualizing data covering international roller coaster rankings and statistics.

-----------------------

### 3. pandas aggregation functions: ["Jeopardy Game"](./statistics_and_visualisation/jeopardy_game/jeopardy_project.ipynb)

The goal of this project is to investigate a dataset of "Jeopardy" game using pandas methods and write several aggregate functions to find some insights. 

-----------------------

### 2. SQL queries (SQLite): ["Marketing Data Analysis"](./sql/funnels/marketing_analysis.ipynb)

The goal of this project is getting customer related insights by:
- applying *usage funnel* marketing model;
- conducting A/B test;
- calculating one of the most important marketing metrics - *churn rate*;
- ascertaining *first- and last-touch attribution* sources; 

using SQL queries on different Codecademy.com datasets.

-----------------------

### 1. Python: ["Medical Insurance"](./python/medical_insurance.ipynb)

The goal of this project is to analyze various attributes within **insurance.csv**, using python, to learn more about the patient information in the file and gain insight into potential use cases for the dataset.   
