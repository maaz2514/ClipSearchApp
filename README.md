# Text2ImageSearch


Brought to you by the Creators of 'Hello World.py'  .... The Text2ImageSearch! 
## Architecture
The Text2ImageSearch implementation follows the following architecture:

* CLIP (Contrastive Language-Image Pre-Training) Model:

    Utilizes the CLIP model to generate image and query (text) embeddings.
* Qdrant Vector Search:

    Implements Qdrant vector search for efficient retrieval of relevant images based on embeddings.
* Streamlit Application:

    Serves the entire system with a Streamlit application, providing an interactive interface for users to perform text-to-(multi)image searches.

## Dataset 
The dataset used for this project comprises advertisement images. This dataset includes advertisements from various countries and in various languages. The total dataset size is 11106. An exploratory evaluation of the dataset has been conducted in the `dataExploration.ipynb` script.

### Installation
* Clone the repository
    ```
    git clone https://github.com/maaz2514/Text2ImageSearch.git
    ```
* Create virtual environment and download image dataset. The dataset can be found at `image_dataset`
    ```
    source setup.sh
    ```
### Run Qdrant
* Pull qdrant docker image 
    ```
    docker pull qdrant/qdrant
    ```
* Run qdrant 
    ```
    docker run -d -p 6333:6333 qdrant/qdrant
     ```
     `-d` to run container in detached mode
* Web UI access: http://0.0.0.0:6333/dashboard

#### Run Text2ImageSearch Implementation
* To use the Text2ImageSearch Implementation, run the following command:

    ```
    python main.py 
    ```
    To use saved image embeddings instead of generating, run:

    ```
    python main.py --load
    ```

#### Streamlit App Interface
The search system uses a steamlit application to interact and display results.

* Under 'Enter your query', type the query to search results for.

* No. of images to display for your query can be adjusted in 'Number of images to show'. (Min 1 and Max 10 images).

* Images related to the query will be displayed under ' Search Results'

![image info](query_eval/evaluation images/app.png)


## Scripts Overview
* `embedder.py`: Contains the `Embedder` class, which generates image and text embeddings using the CLIP model.
* `qdrantClientUpload.py`: Contains the `UploadQdrant` class, responsible for creating a collection of the image embeddings and uploading it to Qdrant.
* `app.py`: This script is the Streamlit app script, which serves as the frontend for interacting with the Text2ImageSearch system.
* `main.py`: This script orchestrates the execution of the above scripts.

## Query Evaluation
The results of query evaluations on the Text2ImageSearch can be found in the `query_eval/queryEvaluation.md` file. This document provides insights into the performance of the search system across various query types and scenarios.

## Additional Improvements
while the implemented Streamlit app interface prioritizes simplicity, there remains room for enhancing the user experience to ensure higher quality interactions.
