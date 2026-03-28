# group8
#Customer Churn Prediction and Customer Segmentation for UK Online Retail Using Data Mining
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load Dataset
df = pd.read_csv('online_retail_II.csv', encoding='latin1')

