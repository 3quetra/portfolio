# Portfolio Projects

Check my [CV](https://www.dropbox.com/scl/fi/4nrwrso76s7j9x76jnrxh/CV_Data_Analyst_Julia_Bukato.pdf?rlkey=wgn8wqco1mlt2ez451m0zwzbk&dl=0). 

This repository contains my Portfolio Projects on various data analysis and data science topics — from python practice and data visualisation to machine learning and natural language processing.


### 51. EDA + Machine Learning: Cancer Diagnosis Prediction (Datathon 2024)

- `Objective`: predict whether breast cancer will be diagnosed within 90 Days.
- `Challenge`: there is an underling discrimination within diagnosis process. 
- `Method`: Lasso Regression for feature selection, mix of classification models for target prediction. TBD
- [Exploratory Data Analysis](./wids_datathon_2024/eda_cancer_data.ipynb).
- [Feature Selection: Approach I](./wids_datathon_2024/feature_selection.ipynb): devide by logical categories, remove highly correlated features and train with simple models: Random Forest, SVM and KNN. 
- [Feature Selection: Approach II](./wids_datathon_2024/feature_selection_2.ipynb): devide by locations, remove highly correlated features and train with boosting models: Light GBM, CatBoost, XGBoost, plus one Deep Learning model: TabNet. 
- `Dataset`:  12 K data points. real-world evidence dataset from Health Verity (HV), one of the largest healthcare data ecosystems in the US as well as addition data from U.S. Postal Service™, U.S. Census Bureau, National Weather Service, American Community Survey, the IRS, NASA/Columbia University.
- `Source`: The WiDS Datathon 2024 (Worldwide).  

-----------------------

### 50. Deep Transfer Learning: [Tender Notice Classification](https://github.com/Ugzuzg/HiTIN)

- `Objective`: develop a model that can accept a tender notice and suggest automatically the most fitting CPV codes.
- `Challenge`: create a model that will classify correctly within the hierarchy with 9000 nodes.
- `Method`: Hierarchy-aware Tree Isomorphism Network for Hierarchical Text Classification combined with EUBERT.
- `Dataset`: ~ 3 mln. Tender descriptions. 
- `Source`: https://ted.europa.eu/en/
- `Applications`: To make it easier to match offer and demand, all public orgs in the EU are required to publish their tender notices with a standard category accompanied by CPV code, defining what they want to buy as precisely as possible. Correct CVP helps potential providers can subscribe to be informed as soon as someone wants something they can sell.
- `Resourses`: [Hierarchy-aware Tree Isomorphism Network for Hierarchical Text Classification](https://arxiv.org/pdf/2305.15182.pdf)

-----------------------

### 49. Statistical Analysis and Visualisation: [Analyzing COVID-19 Impact: Did Vaccination Affect Elderly Mortality in US?](./educational_case_studies/statistics_and_visualisation/nursing_homes_US/nursing_homes_data_analysis_US.ipynb)

- [Project Presentation](./educational_case_studies/statistics_and_visualisation/nursing_homes_US/Analyzing%20COVID-19%20Impact%20Did%20Vaccination%20Affect%20Elderly%20Mortality%20.pdf)
- `Objective`: analyze if COVID-19 vax affected elderly mortality.
- `Results`: T-Statistic: 7.57(SE) > 1% significance threshold - confident null hypothesis rejection. Vaccination appears to have significantly reduced the number of death cases.
- `Data`: COVID-19 reports from nursing homes to CDC's NHSN COVID-19 Facility Module. 
- [Data Collection Methodology](./educational_case_studies/statistics_and_visualisation/nursing_homes_US/covid-19-nursing-home-methodology.pdf)
- `Volume`:  over 2.5 million data points.
- `Coverage`: multiple nursing homes across U.S. states and some islands.
- `Period`: May 24, 2020, to August 27, 2023.
- `Source`: data.CMS.gov (U.S.).

-----------------------

### 48. Statistical Analysis and Visualisation: [Price Runner](./swedish_companies_real_cases/Price_Runner_data_analysis/eda_price_runner.ipynb)

- [Dashboard](https://redash.purpleowling.com/public/dashboards/8LunPkrjwN2NkAuRBmcLHEOG4XhQ09cI3DNNjoXK?org_slug=default) 
- `Objective`: Analyze the overall and unit economics of product creation by considering clicks and click revenues for a given set of products. Present the insights as well as the methodology.
- `Potential Business Impact`: optimize product strategies, enhance marketing efforts, refine resource allocation.
- `3 Datasets`: products, hierarchy, clicks.
- `Source`: Price Runner (Sweden).
- `Results`: The analysis reveals a stable click pattern and correlated cost per click. Product age lacks a clear impact on revenue, but sustained click growth over time is observed, influenced by new product additions and events causing spikes in sales. However, a long-term decline in click numbers is evident.

-----------------------

### 47. Supervised Machine Learning (Time-Series Data Prediction): [Electricity Consumption Forecast (SE)](./swedish_companies_real_cases/Energy_prediction/energy_prediction.ipynb)

- [Project Presentation](./swedish_companies_real_cases/Energy_prediction/Electricity%20Consumption%20Forecast%20(SE).pdf)
- `Objective`: Forecast daily electricity consumption in Sweden for 2022.
- `Potential Business Impact`: more accurate budgeting and potential cost savings. 
- `Final Model R^2 Score`: `0.96`
- `Final Model MAE`: `9.94`
- `Dataset`: ~2K rows × 8 columns.
- `Assumption`: Accurate forecast for the following day is available.
- `Features`: Temperature data (Stockholm, Gothenburg, Lund), Holidays, GWh lags for 7 days.
- `Source`: Vattenfall - Swedish multinational power company. 

-----------------------

### 46. Deep Learning (Image Recognition): ['Deep Transfer Learning for Land Cover Classification'](./educational_case_studies/ml/deep_learning/land_image_classification/land_usage_classifier.ipynb)

- [Application](https://land-cover.purpleowling.com/) 
- [Project Presentation](./educational_case_studies/ml/deep_learning/land_image_classification/Deep%20Transfer%20Learning%20for%20Land%20Cover%20Classification.pdf). 
- `Objective`: develop land cover classifier trained on satellite images.
- `Method`: transfer learning with Wide Residual Networks (`50` CNNs) on the Land Cover (EuroSAT) dataset.
- `Results`: `95%` accuracy with `0.03` loss on test data.
- `Optimizations`: boost efficiency with gradient clipping, adaptive learning rates, and data augmentation.
- `Dataset`: `27,000` labeled images from Sentinel-2 satellite, data obtained via copernicus.eu (EU project).
- `Applications`: maps correction, urban planning, natural catastrophe prediction, precision agriculture.
- `Resourses`: ["Deep Transfer Learning for Land Use and Land Cover Classification: A Comparative Study"](https://www.mdpi.com/1424-8220/21/23/8083)

-----------------------

### 45. EDA (Time-Series Data, Data Scraping): ['EU Energy Generation vs Consumption'](./educational_case_studies/statistics_and_visualisation/EU_energy/eda_EU_energy_handl_comp.ipynb)

- [Project Presentation](./educational_case_studies/statistics_and_visualisation/EU_energy/Comparing%20Energy%20Generation%20and%20Consumption%20in%20EU%20Nations.pdf)
- `Objective`: Comparative Analysis of Energy Generation and Consumption done by **Sweden**, **Germany** and **Poland** during 8 recent years.
- `Results`: 
    - Sweden: dominated by nuclear and renewables.
    - Poland: transitioning from fossil fuels, imports energy.
    - Germany: balanced fossil and renewables, high volume (~7M MW).
    - Consumption: seasonal with spikes (Sweden in 2019 and in 2015).
    - Generation vs. Consumption: surplus (Sweden, Germany), deficit (Poland)
- `Source`: free public datasets dedicated to energy handling within Europe, scraped from https://transparency.entsoe.eu. 

-----------------------

### 44. Supervised Machine Learning (Gradient Boosting on Decision Trees): ['Predicting Extreme Weather Events using Gradient Boost'](/wids_datathon_2023/catboost.ipynb) (Datathon 2023)

- `Objective`: Predicting Extreme Weather Events. 
- `Result`: Using results of EDA, facing challenges like temperature increase in test set due to climate change. Tried to mitigate data drift using Adversarial Validation. Utilized CatBoost for prediction, known for resilience to outliers. Employed Bayesian Optimizer for hyperparameter optimization. 
- `Source`: The WiDS Datathon 2023 (Worldwide).  

-----------------------

### 43. Clusterization (Time-Series Data, K-Means, UMAP): ['Climate Patterns Insight'](./wids_datathon_2023/clustering.ipynb) (Datathon 2023)
- [Project Presentation](./wids_datathon_2023/Climate%20Patterns%20Insight.pdf) 
- `Objective`: analyze climate data using K-Means and UMAP clustering for insights.
- `Results`: 5 Climate patterns, that match Köppen System, have been found: 
    - Cold Semi-Arid: cold winters, limited rain.
    - Mediterranean-Influenced: warm-summer humid continental climate, wet winters.
    - Humid Subtropical: hot summers, mild winters, rainfall year-round.
    - Temperate Oceanic: mild, humid, cool summers.
    - Hot-Summer Continental: hot summers, cold winters, seasonal rain, potentially exceeding year-round rainfall.
- `Source`: WiDS Datathon 2023 (Worldwide).

-----------------------

### 42. EDA (Time-Series Data): ['Exploratory Analysis of Climate Data'](./wids_datathon_2023/eda_temp_prediction.ipynb) (Datathon 2023)

- `Objective`: Conduct Exploratory Data Analysis (EDA) on the training_data.csv and test_data.csv from WiDS Datathon 2023. Discover patterns, relationships, and gain insights into the datasets before further modeling.

- `Result`: EDA insights will guide subsequent modeling and analysis by providing a thorough understanding of the datasets, helping formulate relevant questions, and identifying areas for investigation.

- `Source`: WiDS Datathon 2023 (Worldwide).

-----------------------

### 41. Supervised Machine Learning (Different Models, Gradient Boosting on Decision Trees): [Cars’ CO2 Emissions Prediction](./educational_case_studies/ml/supervised_learning/models_mix/co2_car_emissions/co2_car_emissionsdone.ipynb):

- [Project Presentation](./educational_case_studies/ml/supervised_learning/models_mix/co2_car_emissions/Cars’%20CO2%20Emissions%20Prediction.pdf)
- `Goal`: develop a model to predict CO2 emissions using cars’ characteristics data.
- `Result`: Use formula, when Fuel Consumption is present, it yields the best prediction of CO2. 
- `Tools and Techniques`: different models, Grid Search, Cross Validation.
- `Dataset`: ~7K data points.
- `Features`: 6 numerical, 5 categorical.
- `Source`: Canadian Government Portal (compiled on Kaggle). 

----------------------

### 40 Statistical Analysis and Visualisation (Redash): [Data Analysis and Visualisation for Dr.Bike](./swedish_companies_real_cases/Dr.bike/dr_bike.ipynb)

- [Dashboard](https://redash.purpleowling.com/public/dashboards/V8Z2MkmPsKKwsxtXMrgK9xVSyi8xJ54yDuRNmkYT?org_slug=default)
- `Objective`: - Objective: conduct exploratory data analysis and visualization for Stockholm's bicycle workshop: Dr.Bike.
- `Result`: The most impactfull result for the owner was that returning customers are from Täby and Danderyd. Details are available on the dashboard.
- `Source`: anonymised dataset from Dr.Bike.
----------------------

### 39. Deep Learning (RNNs & LSTMs): ['Temperature Forecast'](./educational_case_studies/ml/deep_learning/time_series/temp_prediction.ipynb)

- `Objective`: Compare the performance of two deep learning algorithms, Tensorflow's stacked layers of Simple Recurrent Neural Network (RNN) and Long-Short-Term Memory Networks (LSTMs), on time-series data. Evaluation metrics include Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

- `Result`: The project aims to provide insights into the relative performance of the mentioned deep learning algorithms on time-series data, facilitating the selection of the most suitable model based on MAE and RMSE metrics.

-----------------------

### 38. Unsupervised Machine Learning (K-Means++): ["Masculinity Survey Insights"](./educational_case_studies/ml/unsupervised_learning/k_means/masculinity_survey/clustering_masculinity_survey.ipynb)

In this project, we will be investigating the way people think about masculinity by applying the **K-Means++** algorithm to data from  <a href="https://fivethirtyeight.com/" target = "_blank">FiveThirtyEight</a>. FiveThirtyEight is a popular website known for their use of statistical analysis in many of their stories.

-----------------------

### 37. Supervised Machine Learning (Ensemble Technique): ["OKCupid Dating App"](./educational_case_studies/ml/supervised_learning/ensemble/dating_app_profiles_analysis.ipynb)

The goal of this project is to find out whether it's possible to predict person's age and religious views via machine learning models, with sufficient accuracy, based on dating app profiles' data from OKCupid.  We are going to evaluate accuracy of different supervised models separately as well as create ensembles and assess resulting accuracy's shifts. Dataset is provided by Codecademy.com.

-----------------------

### 36. NLP: ["Movie Reviews"](./educational_case_studies/nlp/sent_analysis/nlp_portfolio_project.ipynb)

The goal of this project is to get insight into IMDB movie reviews on different layers using various NLP techniques and libraries. A dataset that we are going to use is meant mainly for **binary sentiment classification** and performing this particular type of analysis will, indeed, be our primary goal, but additionally, we'll **use POS tagging and chunking to find commonly used noun and verb phrases** and **find the most common movie topics users wrote about (topic modelling using LDA)**. 

----------------------

### 35. SQL (Windows Functions): ["Climate Change"](./educational_case_studies/sql/windows_functions/climate_change.ipynb )

The goal of this project is to practice SQL windows functions, ascertaining different climate change insights within "Global Land and Ocean Temperatures" dataset in the process.

----------------------

### 34. Unsupervised Learning (K-Means++): ["Handwriting Recognition"](./educational_case_studies/ml/unsupervised_learning/k_means/handwriting_recognition/k_means_handwriting_rec.ipynb)

In this project, we will be using **K-Means++** clustering algorithm in scikit-learn inplementation on sklearn digits dataset to cluster images of handwritten digits.

----------------------


### 33. Supervised Machine Learning (Ensemble Technique - Random Forest): ["Predicting Income"](./educational_case_studies/ml/supervised_learning/random_forest/forest_income_project.ipynb)

In this project, we will be using an ensemble machine learning technique - **Random Forest**. We are going to compare its performance with **Dicision Tree** algorithm, on a dataset containing [census information from UCI’s Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Adult/). By using census data with a Random Forest and Dicision Tree, we will try to predict whether a person makes more than $50,000.

----------------------

### 32. Supervised Machine Learning (Decision Trees): ["Predict Continent and Language"](./educational_case_studies/ml/supervised_learning/trees/trees_find_flag.ipynb)

In this project, we’ll use **Decision Trees** to predict the continent a country is located on, and its language based on features of its flag and other country's properties. For instance, some colors are good indicators as well as the presence or absence of certain shapes could give one a hint. We’ll explore which features are the best to use and will create several Decision Trees to compare their results. The [**Flags Data Set**](https://archive.ics.uci.edu/ml/datasets/Flags) used in this project is provided by UCI’s Machine Learning Repository.

----------------------

### 31. Supervised Machine Learning (Support Vector Machines): ["Predict Baseball Strike Zones"](./educational_case_studies/ml/supervised_learning/svm_models/baseball_strike_zones.ipynb)

In this project, we will use a **Support Vector Machines**, trained using a [`pybaseball`](https://github.com/jldbc/pybaseball) dataset, to find strike zones decision boundary of players with different physical parameters.

----------------------

### 30. Supervised Machine Learning (Naive Bayes): ["Newsgroups Similarity"](./educational_case_studies/ml/supervised_learning/naive_bayes/newsgroups_simularity.ipynb)

In this project we will apply Scikit-learn’s **Multinomial Naive Bayes Classifier** to Scikit-learn’s example datasets to find which category combinations are harder for it to distinguish. We are going to achieve that by reporting the accuracy of several variations of the classifier that were fit on different categories of newsgroups.

----------------------

### 29. Supervised Machine Learning (Logistic Regression): ["Chances of Survival on Titanic"](./educational_case_studies/ml/supervised_learning/logistic_regression/logistic_regression_titanic.ipynb)

The goal of this project is to create a **Logistic Regression Model** that predicts which passengers survived the sinking of the Titanic, based on features like age, class and other relevant parameters.

-----------------------

### 28. Supervised Machine Learning (K-Nearest Neighbor classifier and others): ["Twitter Classification"](./educational_case_studies/ml/supervised_learning/k_nearest_neighbours/twitter_classification.ipynb)

The first goal of this project is to predict whether a tweet will go viral using a **K-Nearest Neighbour Classifier**. The second is to determine whether a tweet was sent from New York, London, or Paris using Logistic Regression and Naive Bayes classifiers with vectorization of different sparsity levels applied to their features. To even out gaps between number of tweets in different cities we'll apply text augmentation by BERT.

-----------------------

### 27. Supervised Machine Learning (Multiple Linear Regression): ["Tendencies of Housing Prices in NYC"](./educational_case_studies/ml/supervised_learning/linear_regression/multiple_linear_regression/prices_tendencies_for_housing.ipynb)

The goal of this project is to get an insight to the range of factors that influence NYC apartments' price formation, predict prices for several random apartments based on these insights with the help of **Multiple Linear Regression Model** and visualise some results using 2D and 3D graphs.

-----------------------

### 26. Supervised Machine Learning (Simple Linear Regression): ["Honey Production"](./educational_case_studies/ml/supervised_learning/linear_regression/simple_linear_regression/honey_production.ipynb)

The goal of this project is predicting honey production during upcoming years (till 2050) using **Simple Linear Regression Model** and some visualizations.

-----------------------

### 25. NLP, Feature Modeling (tf-idf): ["News Analysis"](./educational_case_studies/nlp/language_quantification/news_analysis/language_quantification.ipynb)

In this project we will use **"term frequency-inverse document frequency"** (tf-idf) to analyze each article’s content and uncover the terms that best describe each article, providing quick insight into each article’s topic.

-----------------------

### 24. NLP, Feature Modeling (Word Embeddings): ["U.S.A. Presidential Vocabulary"](./educational_case_studies/nlp/language_quantification/presidential_vocabulary/us_presidential_vocabulary.ipynb)

The goal of this project is to analyze the inaugural addresses of the presidents of the United States of America using **word embeddings**. By training sets of word embeddings on subsets of inaugural addresses, we can learn about the different ways in which the presidents use language to convey their agenda.

-----------------------

### 23. NLP, Topics Analysis (Chunking): ["Discover Insights into Classic Texts"](./educational_case_studies/nlp/language_parsing/language_parsing.ipynb)

The goal of this project is to discover the main themes and some other details from the two classic novels: Oscar Wilde’s **"The Picture of Dorian Gray"** and Homer’s **"The Iliad"**.  To achieve it, we are going to use `nltk` methods for preprocessing and creating Tree Data Structures, after which, we will apply filters to those Structures to get some desired insights.

-----------------------

### 22. NLP (Sentiment Analysis), Supervised Machine Learning (Ensemble Technique): ["Voting System"](./educational_case_studies/nlp/voting_system/nltk_scikitlearn_combined.ipynb)

The goal of this project is to create a voting system for *bivariant sentiment analysis* of any type of short reviews. To achieve this we are going to combine **Naive Bayes** algorithm from `nltk` and similar algorithms from `scikit-learn`. This combination should increase the accuracy and reliability of the confidence percentages.

-----------------------

### 21. NLP, Sentiment Analysis (Naive Bayes Classifier): ["Simple Naive Bayes Classifier"](./educational_case_studies/nlp/naive_bayes_classifier/naive_bayes_classifier.ipynb)

The goal of this project is to build a simple **Naive Bayes Classifier** using `nltk toolkit`, and after that: train and test it on Movie Reviews corpora from `nltk.corpus`.

-----------------------

### 20. Analysis via SQL: ["Gaming Trends on Twitch"](./educational_case_studies/sql/twitch_data_extraction_and_visualization/twitch_data_extraction_and_visualisation.ipynb)

The goal of this project is to analyse gaming trends with SQL and visualise them with Matplotlib and Seaborn. 

-----------------------

 ### 19. Statistical Analysis and Visualisation: ["Airline Analysis"](./educational_case_studies/statistics_and_visualisation/airline_analysis/airline_analysis.ipynb)

The goal of this project is to guide airline company clients' decisions by providing summary statistics that focus on one, two or several variables and visualise its results. 

-----------------------

### 18. Statistical Analysis and Visualisation: ["NBA Trends"](./educational_case_studies/statistics_and_visualisation/associations_nba/nba_trends.ipynb)

In this project, we’ll analyze and visualise data from the NBA (National Basketball Association) and explore possible associations.

-----------------------

### 17. Statistical Analysis and Visualisation: ["Titanic"](./educational_case_studies/statistics_and_visualisation/quant_and_categorical_var/titanic.ipynb)

The goal of this project is to investigate whether there are some correlations between the different aspects of physical and financial parameters and the survival rates of the Titanic passengers.

-----------------------

### 16. Data Transformation: ["Census Data Types Transformation"](./educational_case_studies/statistics_and_visualisation/census_datatypes_transform/census_datatypes_transform.ipynb)

The goal of this project is to use pandas to clean, organise and prepare recently collected census data for further usage by machine learning algorithms.

-----------------------

### 15. Data Transformation: ["Company Transformation"](./educational_case_studies/statistics_and_visualisation/data_transformation/company_transformation.ipynb)

The goal of this project is to apply data transformation techniques to better understand the company’s data and help to answer important questions from the management team.


-----------------------

### 14. Statistical Analysis and Visualisation: ["Roller Coasters"](./educational_case_studies/statistics_and_visualisation/roller_coasters/roller_coaster.ipynb)
The goal of this project is visualizing data covering international roller coaster rankings and statistics.

-----------------------

### 13. pandas aggregation functions: ["Jeopardy Game"](./educational_case_studies/statistics_and_visualisation/jeopardy_game/jeopardy_project.ipynb)

The goal of this project is to investigate a dataset of "Jeopardy" game using pandas methods and write several aggregate functions to find some insights. 

-----------------------

### 12. SQL queries (SQLite): ["Marketing Data Analysis"](./educational_case_studies/sql/funnels/marketing_analysis.ipynb)

- `Objective`: Extract customer insights through usage funnel model, A/B tests, churn rate calculation, and first- and last-touch attribution. Optimize marketing strategies.
- `Sources`: datasets from Warby Parker, Codeflix, and CoolTShirts obtained by Codecademy.com .
- `Usage Funnels`: Construct Quiz and Home Try-On funnels for Warby Parker, analyzing user journeys and conversion rates.
- `Results`: A/B Test insights for Warby Parker recommend offering 5 pairs for home try-on. Codeflix should focus on expanding user base in segment 30. Key campaigns, like weekly-newsletter, bring high purchases for CoolTShirts.

-----------------------

### 11. Python: ["Medical Insurance"](./educational_case_studies/python/medical_insurance.ipynb)

The goal of this project is to analyze various attributes within **insurance.csv**, using python, to learn more about the patient information in the file and gain insight into potential use cases for the dataset.   

----------------------

### 10. Python Problems: ['Binary Search'](./educational_case_studies/algorithms_and_data_structures/binary_search.ipynb)
The goal of this project is to practice implementation of binary search and solve some problems using it. 
- Problem: Sharing a chocolate bar fairly among 11 people.
- Objective: Maximize the sweetness of the piece with minimal sweetness.

Short [**presentation**](./educational_case_studies/algorithms_and_data_structures/Optimizing%20Chocolate%20Sharing%20with%20Binary%20Search.pdf).

-----------------------

### 9. Python Problems: ['Arrays and strings'](./educational_case_studies/algorithms_and_data_structures/arrays_and_strings.ipynb)

"In terms of algorithm problems, arrays (1D) and strings are very similar: they both represent an ordered group of elements. Most algorithm problems will include either an array or string as part of the input, so it's important to be comfortable with the basic operations and to learn the most common patterns". (leetcode.com)

The goal of this project is to get comfortable with basic python operations that involve arrays and strings.

-----------------------

### 8. Explorative Data Analysis (EDA): ["Diagnosing Diabetes"](./educational_case_studies/statistics_and_visualisation/eda_diagnosing_diabetes/eda.ipynb)

The goal of this project is to explore data that looks at how certain diagnostic factors affect the diabetes outcome of women patients. We are going to use EDA skills to help inspect, clean, and validate the data.

-----------------------

### 7. Visualisaton using boxplot: ["Healthcare in Different States"](./educational_case_studies/statistics_and_visualisation/healthcare_in_different_states/healthcare_in_different_states.ipynb)

In this project, we will use boxplots to investigate the way hospitals in different states across the United States charge their patients for medical procedures.

-----------------------

### 6. Visualisation using histogram: ["Traveling to Acadia"](./educational_case_studies/statistics_and_visualisation/traveling_to_acadia/traveling_to_acadia.ipynb)

The goal of this project is to inform customers of a travel agency about the best time to visit Acadia to see flowers in bloom. We will use flower bloom data, and flight information to create visualisation and recommend the best time of year for someone to make a trip to Maine.

-----------------------

### 5. Analysis using statistical instruments: ["Life expectancy by country"](./educational_case_studies/statistics_and_visualisation/life_expectancy_by_country/life_expectancy_by_country.ipynb)

The goal of this project is to investigate a dataset containing information about the average life expectancy in 158 different countries and, using statistical instruments, analyse an example country's location on a GDP scale based on the average life expectancy of this country.

-----------------------

### 4. Data wrangling, tidying and visualisation: ["Cleaning US Census Data"](./educational_case_studies/statistics_and_visualisation/us_census_data/cleaning_us_census_data.ipynb)

The goal of this project is to make US Census data scalable and repeatable, converting data from multiple CSV files to pandas' DataFrames, as well as to collect some interesting insights from it and create informative visualisations as a result. 

-----------------------


### 3. WebScraping with Beautiful Soup: ["Chocolate bars"](./web_scraping/chocolate_scraping.ipynb)

The goal of this project is to analyze and manipulate data displayed in a table on a website: `https://content.codecademy.com/courses/beautifulsoup/cacao/index.html`, using BeautifulSoup and pandas that should help us transform this webpage into a DataFrame.

-----------------------

### 2. Visualisation using Seaborn: ["Visualizing World Cup Data"](./educational_case_studies/statistics_and_visualisation/visualizing_world_cup_data/visualizing_fifa_data.ipynb)

The goal of this project is using Seaborn explore data from the Fifa World Cup from 1930-2014 to analyze trends and discover insights about football game.

-----------------------

### 1. Building Funnel via pandas: ["Page Visits"](./educational_case_studies/statistics_and_visualisation/page_visits_funnel/page_visits_funnel.ipynb)

The goal of this project is to analyze data on visits to a website of Cool T-Shirts Inc by building a funnel via pandas.




    



