# Machine Learning Projects by Vishal Nair

This repository contains a collection of machine learning projects developed by Vishal Nair. The projects cover various aspects of machine learning, including sentiment analysis, price prediction, customer segmentation, and regression modeling.

## Projects

1. [Sentiment Analysis of IMDB Movie Reviews](#sentiment-analysis-of-imdb-movie-reviews)
2. [Car Price Prediction](#car-price-prediction)
3. [Customer Segmentation using K-Means Clustering](#customer-segmentation-using-k-means-clustering)
4. [Boston Housing Model](#boston-housing-model)

---

## Sentiment Analysis of IMDB Movie Reviews

**File:** `sentimentAnalysis.ipynb`

### Project Description
This project performs sentiment analysis on the IMDB movie reviews dataset. The goal is to classify the reviews as positive or negative.

### Data
- **Dataset:** IMDB_Dataset.csv
- **Columns:** `review` (text), `sentiment` (positive/negative)

### Models Used
- Logistic Regression
- Naive Bayes
- Linear Support Vector Classification (SVC)

### Machine Learning Techniques
- Text preprocessing: Tokenization, stop words removal, stemming
- Feature extraction: TF-IDF Vectorization
- Model training and evaluation

### Metrics Used
- Accuracy
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)

### Libraries
- pandas, matplotlib, re, BeautifulSoup, nltk, plotly, scikit-learn

---

## Car Price Prediction

**File:** `Car_Price_Prediction.ipynb`

### Project Description
This project aims to predict the selling price of cars based on various features such as present price, kilometers driven, fuel type, seller type, transmission, and owner.

### Data
- **Dataset:** car data.csv
- **Columns:** `Car_Name`, `Year`, `Selling_Price`, `Present_Price`, `Kms_Driven`, `Fuel_Type`, `Seller_Type`, `Transmission`, `Owner`

### Models Used
- Linear Regression
- Lasso Regression

### Machine Learning Techniques
- Data encoding and preprocessing
- Model training and evaluation
- Data visualization

### Metrics Used
- R-squared Error

### Libraries
- pandas, matplotlib, seaborn, scikit-learn

---

## Customer Segmentation using K-Means Clustering

**File:** `Customer_Segmentation_using_K_Means_Clustering.ipynb`

### Project Description
This project performs customer segmentation using K-Means Clustering. The goal is to group customers into distinct segments based on their annual income and spending score.

### Data
- **Dataset:** Mall_Customers.csv
- **Columns:** `CustomerID`, `Gender`, `Age`, `Annual Income (k$)`, `Spending Score (1-100)`

### Models Used
- K-Means Clustering

### Machine Learning Techniques
- Data preprocessing
- Finding the optimal number of clusters using the Elbow method
- Training the K-Means model
- Visualizing customer groups and cluster centroids

### Metrics Used
- Within-Cluster Sum of Squares (WCSS)

### Libraries
- pandas, matplotlib, seaborn, scikit-learn

---

## Boston Housing Model

**File:** `VishalNair-BostonModel`

### Project Description
This project uses regression techniques to predict house prices in Boston. The dataset includes various features such as crime rate, number of rooms, and property tax rate.

### Data
- **Dataset:** Boston Housing Dataset
- **Columns:** Various features related to housing in Boston

### Models Used
- Linear Regression
- Decision Tree Regression
- Random Forest Regression

### Machine Learning Techniques
- Data preprocessing
- Model training and evaluation
- Hyperparameter tuning

### Metrics Used
- Mean Squared Error (MSE)
- R-squared Error

### Libraries
- pandas, scikit-learn, seaborn, matplotlib

---

## Author

- **Vishal Nair**
- [Email](mailto:v1292002@gmail.com)
- [LinkedIn](https://www.linkedin.com/in/vishal-nair-9a87a1183/)

