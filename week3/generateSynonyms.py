from typing import List, Any

import fasttext


test_tokens = [
    "gift", "camera", "laptop", "battery", "dishwasher", # products,
    "apple", "sony", "barnes & noble", "microsoft", "bosch", # brands,
    "2001", "beats by dr.dre", "playstation2", "ipad2", "integra 500", # models,
    "silver", "pink", "1080p", "lcd", "100$", # attributes 
]

def train_model(input_file: str, min_count: int):
    """Trains fasttext model with given configs, and returns model"""
    # other ws, minn, maxn
    kwargs = {
        "input": input_file, # training data
        "epoch": 50,
        "minn": 0,
        "maxn": 0,
        "dim": 150,
        "minCount": min_count,
        "model": "skipgram"
    }
    model = fasttext.train_unsupervised(**kwargs)
    return model


def get_synonyms(model: Any, test_tokens: List[str]):
    for token in test_tokens:
        print(f"Test word: {token}")
        neighbors = model.get_nearest_neighbors(token, 10)
        for neighbor in neighbors[0:3]:
            # print 3 nearest neighbors
            print(neighbor)
        
if __name__ == "__main__":
    # base_dir = '/workspace/datasets/fasttext/'
    output_dir = "/workspace/datasets/fasttext/title_model"
    model = None
    model_exists = True

    if not model_exists:
        model = train_model(input_file="/workspace/datasets/fasttext/titles.txt", min_count=20)

    model = fasttext.load_model(f"{output_dir}.bin")

    model.save_model(output_dir)
    get_synonyms(model, test_tokens)


#### Results
