## YouTube Playlist Engagement Pattern Analysis
This repository contains code and documentation for the project “Audience Engagement Pattern Analysis on YouTube Videos Using K-Means and DBSCAN Algorithms”. An interactive dashboard application was built using [Streamlit](https://streamlit.io/) to analyze and group videos from any YouTube playlist based on audience engagement patterns.

## Technologies Used
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=seaborn&logoColor=white)](https://seaborn.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![YouTube Data API v3](https://img.shields.io/badge/YouTube%20Data%20API%20v3-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://developers.google.com/youtube/v3)

## Key Features
Dynamic Analysis: Users can input Playlist IDs from any YouTube channel for live analysis.
Automatic Data Preprocessing: Includes outlier removal and feature normalization using StandardScaler.
Feature Engineering: Creating engagement_rate metrics to gain deeper insights.
Clustering with K-Means: Identified 3 main video segments (High, Medium, and Low Performance).
Interactive Visualization: Results are presented in the form of PCA plots, summary tables, and inter-cluster comparison graphs.

## About the Project
Audience engagement analysis on YouTube is often limited to surface metrics such as the number of views, likes, and comments, without delving into deeper engagement patterns. This project aims to overcome these limitations by clustering videos based on audience engagement patterns using K-Means and DBSCAN clustering methods.
Quantitative data (views, likes, comments) from Deddy Corbuzier's YouTube channel (as an initial case study) was collected through YouTube API v3. This data-driven segmentation model can be utilized by content creators for content personalization and by advertisers for more effective targeting.

# Creator
[Wahyu Aji Nusantara](https://www.wahyuaji.cloud)
