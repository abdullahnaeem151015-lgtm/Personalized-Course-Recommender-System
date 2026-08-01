# Personalized Course Recommender System

A machine learning-based course recommendation system that suggests relevant courses based on a learner's completed learning history.

The project combines multiple recommendation techniques and presents them in an interactive Streamlit dashboard where users can compare different recommendation methods, adjust model parameters, and generate personalized course recommendations.

---
## Live Demo ##

https://personalized-course-recommender-system-74bhfr6ufblqnrwrypn5nu.streamlit.app/

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Overview

Finding the right next course can be difficult when learners have many options available.

This project explores different recommendation techniques and compares how each approach recommends courses based on user behavior, course similarity, and learned patterns from historical interactions.

The final result is an interactive web application where users can:

- Select courses they have already completed
- Choose different recommendation methods
- Adjust model parameters
- Generate personalized recommendations
- Compare recommendation results
- View model performance and prediction evidence

---

## Objectives

- Build multiple recommendation models
- Compare collaborative filtering and content-based approaches
- Evaluate different machine learning algorithms
- Provide an easy-to-use recommendation dashboard
- Help learners discover relevant courses based on their learning history

---

# Recommendation Methods

The application includes the recommendation approaches covered throughout the project.

### Content-Based Methods

- Course Similarity
- User Profile

### Collaborative Filtering Methods

- Clustering
- Clustering with PCA
- K-Nearest Neighbors (KNN)
- Non-negative Matrix Factorization (NMF)
- Neural Network Embeddings
- Classification using Embedding Features
- Regression using Embedding Features

---

# Machine Learning Models

Different supervised learning models were trained and compared for recommendation prediction.

Models include:

- Logistic Regression
- Decision Tree
- Support Vector Machine (SVM)
- Random Forest
- Bagging
- AdaBoost
- Gradient Boosting

Hyperparameter tuning was performed using Grid Search with cross-validation to identify better-performing model configurations.

---

# Evaluation Metrics

Model performance was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC (where applicable)
- Cross-validation scores

These metrics help compare how different models perform on the recommendation task.

---

# Streamlit Dashboard

The project includes a fully interactive Streamlit application.

Users can:

### Build Recommendations

- Select completed courses
- Use an existing learner ID
- Choose a recommendation method
- Adjust model parameters
- Generate recommendations

### Compare Recommendation Methods

Users can compare how different recommendation approaches perform using the same learning history.

### Visualize Results

The dashboard displays:

- Recommended courses
- Recommendation scores
- Interactive charts
- Model comparison visuals

### Learn How Recommendations Work

A beginner-friendly section explains:

- How recommendations are generated
- What each recommendation method does
- How model predictions are evaluated
- Strengths and limitations of each approach

### Prediction Evidence

The dashboard also provides:

- Model evaluation metrics
- Performance summaries
- Recommendation confidence information
- Transparent explanation of prediction quality

---

# Project Workflow

1. Load course and interaction datasets
2. Prepare and preprocess data
3. Build recommendation features
4. Train multiple recommendation models
5. Tune model parameters
6. Evaluate model performance
7. Generate personalized recommendations
8. Visualize recommendations through Streamlit

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Streamlit
- Plotly
- Matplotlib

---

# Key Features

- Multiple recommendation techniques
- Interactive Streamlit dashboard
- Hyperparameter tuning
- Machine learning model comparison
- Recommendation score visualization
- Beginner-friendly explanation of model predictions
- Transparent evaluation metrics


---

# Limitations

- Recommendations depend on the available course interaction data.
- Recommendation quality may vary depending on the selected method and learning history.
- Some models require more training time than others.
- Recommendation scores indicate relative relevance rather than guaranteeing a learner will like a course.

---

# Future Improvements

Possible future enhancements include:

- Support for larger course catalogs
- Hybrid recommendation models
- Real-time recommendation updates
- User feedback integration
- Improved recommendation explanations

---

# Author

**Abdullah Naeem**

BS Computer Science Student

Interested in Machine Learning, Data Science, and AI-powered applications.

---

If you found this project helpful or interesting, feel free to explore the code, try the Streamlit application, or share your feedback.
