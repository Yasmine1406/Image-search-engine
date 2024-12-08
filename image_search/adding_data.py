from elasticsearch import Elasticsearch, exceptions
from elasticsearch.helpers import bulk
import pandas as pd
import numpy as np
from elasticsearch.helpers import bulk, BulkIndexError

# Connect to Elasticsearch
es = Elasticsearch("http://localhost:9200/")
print('Connected to ElasticSearch')

# Define the index settings and mappings
try:
    index_name = 'images_data'
    # Define the index settings and mappings
    settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "path": {
                    "type": "text"
                },
                "vector": {
                    "type": "dense_vector",
                    "dims": 4096  
                }
            }
        }
    }

    # Create the index
    if es.indices.exists(index=index_name):
        # Delete the index if it exists
        es.indices.delete(index=index_name)
        print(f"Index '{index_name}' has been deleted.")
    else:
        print(f"Index '{index_name}' does not exist.")

    es.indices.create(index=index_name, body=settings)
    print(f"Index '{index_name}' has been created.")

    # Read data from CSV file
    data = pd.read_csv(r"D:\Tbourbi\ImageSearchEngine\feature_vectors.csv")
    print('Read CSV file')

    # Prepare the data for bulk indexing
    bulk_data = []
    # Loop through the rows of the CSV file
    for _, row in data.iterrows():
        print('Iterating through the rows...')
        # Extract the feature vector 
        feature_vector = row.iloc[:-1].apply(pd.to_numeric, errors='coerce').values.astype(np.float32).tolist()
    
        # Extract the image path 
        image_path = row.iloc[-1]

        # Truncate the vector to 4096 dimensions (if needed)
        if len(feature_vector) > 4096:
            feature_vector = feature_vector[:4096]

        # Create a document to be indexed
        doc = {
            "_index": index_name,
            "_source": {
                "path": image_path,
                "vector": feature_vector
            }
        }
        bulk_data.append(doc)

    # Perform bulk indexing
    if bulk_data:
        try:
            bulk(es, bulk_data, request_timeout=500)
        except BulkIndexError as e:
            for error in e.errors:
                print(error)
        print(f"Successfully indexed {len(bulk_data)} documents.")
    else:
        print("No data to index.")

except exceptions.RequestError as e:
    # Handle any errors related to request issues, such as mapping errors
    error_info = e.info
    print("Mapping-related error info:")
    print(error_info)
