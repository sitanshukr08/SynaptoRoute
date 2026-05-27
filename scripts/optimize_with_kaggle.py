import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from dotenv import load_dotenv

from synaptoroute.encoder import Encoder
from synaptoroute.storage import SQLiteStorage
from synaptoroute.router import AdaptiveRouter
from synaptoroute.models import Route

load_dotenv()

def main():
    # 1. Load the downloaded CSV
    csv_path = 'data/Bitext_Sample_Customer_Service_Training_Dataset.csv'
    df = pd.read_csv(csv_path)
    
    # 2. Filter down to 4 distinct intents
    top_intents = df['intent'].value_counts().head(4).index.tolist()
    df = df[df['intent'].isin(top_intents)]
    
    # 3. Sample 75 utterances per intent (or whatever is available)
    df = df.sample(frac=1, random_state=42).groupby('intent').head(75).reset_index(drop=True)
    
    # 4. Initialize AdaptiveRouter and clear local sqlite
    db_path = os.getenv('SQLITE_DB_PATH', 'data/router_memory.sqlite')
    if os.path.exists(db_path):
        os.remove(db_path)
    
    encoder = Encoder()
    storage = SQLiteStorage(db_path)
    router = AdaptiveRouter(encoder, storage)
    
    # 5. Split data into 80% training / 20% validation
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['intent'])
    
    # 6. Add training utterances to the router
    print(f"Adding training utterances to the router for {len(top_intents)} intents...")
    for intent in top_intents:
        intent_train_texts = train_df[train_df['intent'] == intent]['utterance'].tolist()
        route = Route(name=intent, utterances=intent_train_texts, threshold=0.1)
        router.add_route(route)
        
    train_texts = train_df['utterance'].tolist()
    train_labels = train_df['intent'].tolist()
    
    # 7. Fit thresholds
    print("Fitting thresholds...")
    router.fit_thresholds(train_texts, train_labels)
    
    # 8. Iterate over validation set and compute F1
    print("Evaluating on validation set...")
    val_texts = val_df['utterance'].tolist()
    val_labels = val_df['intent'].tolist()
    
    preds = []
    for text in val_texts:
        match = router(text)
        if match:
            preds.append(match.name)
        else:
            preds.append("None")
            
    f1 = f1_score(val_labels, preds, average='weighted', zero_division=0)
    print(f"Validation Weighted F1-Score: {f1:.4f}")
    
    # 9. Print optimized thresholds
    print("\nOptimized Thresholds:")
    for route_name, route in router._route_map.items():
        print(f"  {route_name}: {route.threshold:.4f}")

if __name__ == "__main__":
    main()
