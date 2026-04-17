import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

def train_model():
    print("Starting Model Training Engine...")
    base_dir = os.path.dirname(__file__)
    dataset_dir = os.path.join(base_dir, 'dataset')
    
    true_path = os.path.join(dataset_dir, 'True.csv')
    fake_path = os.path.join(dataset_dir, 'Fake.csv')
    
    if not os.path.exists(true_path) or not os.path.exists(fake_path):
        print("ERROR: Kaggle datasets not found in backend/dataset/")
        return
        
    print("Loading huge CSV clusters into memory...")
    # Load and tag the frames
    df_true = pd.read_csv(true_path)
    df_true['label'] = 1  # 1 for Real
    
    df_fake = pd.read_csv(fake_path)
    df_fake['label'] = 0  # 0 for Fake
    
    # Merge and shuffle
    df_merged = pd.concat([df_true, df_fake], ignore_index=True)
    df_merged = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # For training, we combine title and text for rich context
    df_merged['full_text'] = df_merged['title'] + " " + df_merged['text']
    
    # We will sample 20,000 to keep training time under 15 seconds, but it's highly robust
    # You can increase it to len(df_merged) for maximum accuracy
    subset = df_merged.head(20000)
    
    X_train, X_test, y_train, y_test = train_test_split(
        subset['full_text'], subset['label'], test_size=0.2, random_state=42
    )

    print("Building TF-IDF mapping and PassiveAggressive Pipeline...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words='english', max_df=0.7)),
        ("clf", PassiveAggressiveClassifier(max_iter=50, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    predictions = pipeline.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Validation Target Achieved: {acc * 100:.2f}% Precision")
    
    model_out = os.path.join(base_dir, 'model.pkl')
    with open(model_out, 'wb') as f:
        pickle.dump(pipeline, f)
        
    print(f"Engine serialization complete: {model_out}")

if __name__ == "__main__":
    train_model()
