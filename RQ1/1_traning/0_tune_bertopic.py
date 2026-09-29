import json
import numpy as np
import os
import logging
import itertools 
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech
from hdbscan import HDBSCAN 
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from bertopic.vectorizers import ClassTfidfTransformer 
import pickle
import csv   
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable Huggingface tokenizers parallelism warnings

import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RQ2_CATEGORIES_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..')) 
sys.path.insert(0, RQ2_CATEGORIES_DIR)
from utilities.rq2_ulities import Rq2Ulities

warnings.filterwarnings("ignore", category=RuntimeWarning)

def setup_logging(file_name):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(file_name, mode="a"),
        ],
    )
    logging.getLogger("py4j").setLevel(logging.WARNING)
    logging.getLogger("pyspark").setLevel(logging.WARNING)
    logging.getLogger("hdbscan").setLevel(logging.WARNING)
    logging.getLogger('gensim').setLevel(logging.WARNING)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def generate_param_combinations(param_ranges):
    """
    Generate all possible combinations of parameters for tuning.
    """
    keys, values = zip(*param_ranges.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]

def configure_topic_model(params, embedding_model):
    """
    Configure the BERTopic model with given hyperparameters.
    """
    umap_model = UMAP(
        n_neighbors=params['n_neighbors'],
        n_components=params['n_components'],
        min_dist=params['min_dist'],
        metric="cosine", # default
        random_state=42
    )  
    hdbscan_model = HDBSCAN(
        min_cluster_size=params['min_cluster_size'],
        min_samples=params['min_samples'],
        cluster_selection_method="eom",
        metric="euclidean",
        gen_min_span_tree=True,
        prediction_data=True
    )
    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 2))
    ctfidf_model = ClassTfidfTransformer(bm25_weighting=True, reduce_frequent_words=True)

    representation_model = {
        "KeyBERT": KeyBERTInspired(),
        "MMR": MaximalMarginalRelevance(diversity=0.1),
        # "POS": PartOfSpeech("en_core_web_sm")
    }

    return BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,
        top_n_words=10,
        calculate_probabilities=True,
        verbose=False
    )
 
def save_cluster_assignments(docs, metadata_list, topics, probs, output_dir):
    """Save cluster assignments with full conversation metadata"""
    os.makedirs(output_dir, exist_ok=True)
    
    results_df = pd.DataFrame({
        "conversation_id": [m["conversation_id"] for m in metadata_list],
        "filename": [m["filename"] for m in metadata_list],
        "turn_number": [m["turn_number"] for m in metadata_list],
        "processed_text": docs,
        "original_content": [m["original_content"] for m in metadata_list],
        "cluster": topics,
        "probability": [max(p) if isinstance(p, np.ndarray) else p for p in probs]
    })
    
    # Save to CSV
    csv_path = os.path.join(output_dir, "cluster_assignments.csv")
    results_df.to_csv(csv_path, index=False)
    
    # Also save as JSON for easier conversation analysis
    json_path = os.path.join(output_dir, "cluster_assignments.json")
    results_df.to_json(json_path, orient="records", indent=2)
    return results_df

# Function to save scores with Iteration and Cluster Count
def save_scores_to_csv(iteration_num, num_clusters, coherence_scores, hdbscan_scores, params, csv_file_path, noise_percentage):
    """
    Save coherence scores, HDBSCAN scores, and corresponding parameters to a CSV file.
    Now includes Iteration Number and Cluster Count as the first two columns.
    """
    header = (
        ["Iteration", "Cluster Count"] +  
        list(params.keys()) +  
        ["c_v", "u_mass", "c_uci", "c_npmi", "relative_validity", 
         "outlier_scores_mean", "outlier_scores_median", 
         "persistence_scores_mean", "persistence_scores_median", "noise_percentage"]
    ) 
    file_exists = os.path.isfile(csv_file_path) 
    
    with open(csv_file_path, mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header) 
        if not file_exists:
            writer.writeheader() 
            
        row = {
            "Iteration": iteration_num,
            "Cluster Count": num_clusters,
            **params, 
            **coherence_scores, 
            **hdbscan_scores, 
            "noise_percentage": noise_percentage
        }
        writer.writerow(row)
        

if __name__ == "__main__":
    import argparse

    # Add argument parsing for parallel runs
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0, help="Start index for parameter combinations")
    parser.add_argument("--end", type=int, default=None, help="End index for parameter combinations")
    args = parser.parse_args()

    # Setup logging
    save_path = "../rq2_results/0_bertopic_30june/"
    os.makedirs(save_path, exist_ok=True)
    setup_logging(file_name=save_path + "0_hyperparameter_tuning.log")

    # Load dataset and embeddings
    folder_path = "../../data/code_chat_fullturn"
    rq2_unit_pro = Rq2Ulities(folder_path)
    docs_with_metadata = rq2_unit_pro.eng_dataset_first_turn_user_only()
    docs = [doc[0] for doc in docs_with_metadata]
    metadata_list = [doc[1] for doc in docs_with_metadata]

    # Generate embeddings
    embeddings, embedding_model = rq2_unit_pro.generate_embeddings(docs)

    # Parameter grid
    param_ranges = {
        "n_neighbors": range(5, 16, 5),
        "min_dist": [0.0],
        "n_components": range(2, 10, 3),
        "min_cluster_size": range(30, 201, 10),
        "min_samples": range(5, 21, 5)
    }
    
    # Generate all parameter combinations
    param_combinations = generate_param_combinations(param_ranges)  
    logging.info(f"Total parameter combinations: {len(param_combinations)}")

    start_iteration = args.start
    end_iteration = args.end if args.end is not None else len(param_combinations)

    # Main tuning loop
    for i, params in enumerate(param_combinations[start_iteration:end_iteration], start=start_iteration):
        logging.info(f"Iteration {i + 1}/{len(param_combinations)} - Testing parameters: {params}")

        # Configure and train BERTopic
        topic_model = configure_topic_model(params, embedding_model)
        topics, probs = topic_model.fit_transform(docs, embeddings)

        # Calculate noise percentage
        num_outliers = np.sum(topics == -1)
        noise_percentage = num_outliers / len(topics)
        
        if noise_percentage > 0.30:
            logging.info(f"Skipping parameters due to high noise ({noise_percentage:.2f} > 30%)")
            continue

        # Save cluster assignments
        model_output_dir = os.path.join(save_path, f"model_iter_{i+1}")
        cluster_df = save_cluster_assignments(docs, metadata_list, topics, probs, model_output_dir)
        
        # Get HDBSCAN scores
        hdbscan_model = topic_model.hdbscan_model
        hdbscan_scores = {
            "relative_validity": hdbscan_model.relative_validity_,
            "outlier_scores_mean": np.mean(hdbscan_model.outlier_scores_),
            "outlier_scores_median": np.median(hdbscan_model.outlier_scores_),
            "persistence_scores_mean": np.mean(hdbscan_model.cluster_persistence_),
            "persistence_scores_median": np.median(hdbscan_model.cluster_persistence_)
        }
        
        # Compute coherence scores
        coherence_scores = rq2_unit_pro.cal_coherence(
            topic_model, 
            docs, 
            coherence_types=["c_v"]
        )

        # Save model and topic info
        rq2_unit_pro.save_model_safetensors(topic_model, topics, docs, probs, model_output_dir)
        
        # Get cluster count
        num_clusters = len(set(topics)) - (1 if -1 in topics else 0)
        
        # Save evaluation metrics
        save_scores_to_csv(
            iteration_num=i+1,
            num_clusters=num_clusters,
            coherence_scores=coherence_scores,
            hdbscan_scores=hdbscan_scores,
            params=params,
            csv_file_path=os.path.join(save_path, "hyperparameters.csv"),
            noise_percentage=noise_percentage
        )

    logging.info("Hyperparameter tuning complete.")

