import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import tldextract

def extract_url_features(url):
    """Extracts numerical and categorical lexical features from a URL string."""
    features = {}
    
    # 1. Length features
    features['url_length'] = len(url)
    features['hostname_length'] = len(tldextract.extract(url).domain)
    
    # 2. Presence of suspicious symbols
    features['has_ip'] = 1 if any(char.isdigit() for char in url.split('/')[0]) else 0
    features['count_at'] = url.count('@')
    features['count_question'] = url.count('?')
    features['count_percent'] = url.count('%')
    features['count_dot'] = url.count('.')
    features['count_hyphen'] = url.count('-')
    
    # 3. Protocol check
    features['is_https'] = 1 if url.startswith('https') else 0
    
    return features

# Dummy data generation / Loading your dataset
# In practice, load a CSV with columns: ['url', 'label'] where label is 1 (phishing) or 0 (legitimate)
def load_data():
    # Sample training data for demonstration (Replace with full CSV load)
    data = {
        'url': [
            'https://www.google.com', 'https://github.com/login', 
            'http://192.168.1.1/update-bank-account-secure', 
            'http://verify-apple-id-support-security.com/login',
            'https://www.amazon.com/gp/help', 'http://paypal-security-update.xyz/signin'
        ],
        'label': [0, 0, 1, 1, 0, 1] # 0 = Legitimate, 1 = Phishing
    }
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = load_data()
    
    # Extract features for all URLs in dataset
    feature_list = [extract_url_features(url) for url in df['url']]
    X = pd.DataFrame(feature_list)
    y = df['label']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    print("Model Accuracy:", accuracy_score(y_test, predictions))
    
    # Save the trained model
    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'model/model.pkl')
    print("Model saved successfully to model/model.pkl")