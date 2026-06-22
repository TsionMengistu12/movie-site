# movie-site

# Overview
This project implements a Hybrid Movie Recommendation System using the MovieLens 100K dataset.

The system combines:
1. Content-Based Filtering
2. Collaborative Filtering
3. Hybrid Recommendation

The goal is to provide personalized movie recommendations by utilizing both movie content information and user rating behavior.

## Dataset
MovieLens 100K Dataset
Files Used:
- u.data
- u.item
Dataset Statistics:
- 100,000 ratings
- 943 users
- 1,682 movies
- 
## Technologies Used
Programming Language:
- Python

Libraries:
- Pandas
- NumPy
- Scikit-Learn
- Surprise
- Matplotlib
- Streamlit
- Pickle

## Methodology

### Content-Based Filtering
Movie titles and genres are combined into a text representation.
TF-IDF Vectorization is used to convert textual information into numerical vectors.
Cosine Similarity is computed to determine similarity between movies.

### Collaborative Filtering
Collaborative filtering is implemented using Singular Value Decomposition (SVD) from the Surprise library.
The model learns latent relationships between users and movies from rating data.

### Hybrid Recommendation
The final recommendation score is computed as:
Hybrid Score = 0.7 × SVD Prediction + 0.3 × Content Similarity
Movies are ranked according to this hybrid score and the top recommendations are returned.

## Evaluation Metrics
The system was evaluated using:
- RMSE
- MAE
- Precision@5
- Recall@5
- Coverage

Sample Results:
- RMSE: 0.93
- MAE: 0.73
- Precision@5: 0.72
- Recall@5: 0.65
- Coverage: 63.2%

## Running the Project

### Install Dependencies

pip install -r requirements.txt

### Train the Model

python train.py

### Run the Application

python -m streamlit run app.py

## Features

- Personalized movie recommendations
- Hybrid recommendation algorithm
- Interactive Streamlit interface
- Recommendation score visualization
- Recommendation ranking table

## The Project look
<img width="1828" height="872" alt="image" src="https://github.com/user-attachments/assets/92eb995f-bab8-4f1f-be0c-addb0fb326cb" />
<img width="1608" height="593" alt="image" src="https://github.com/user-attachments/assets/70dccdf6-412e-4ce4-8b0b-cb871d30717d" />
<img width="1096" height="572" alt="image" src="https://github.com/user-attachments/assets/466ba79d-3a9c-450d-ba04-b2d4bc745c79" />

## The Evaluation.py running
<img width="517" height="170" alt="image" src="https://github.com/user-attachments/assets/2fa63a82-90b8-48ba-9930-df50284964db" />


