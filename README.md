# SIGMOD PROJECT

#### Team members:
- Andrea Veneziano
- Georgios Fotiadis
- Gerald Sula

## Code information

You can find all of our code in the **src/** folder.
Following are the explanation of what each notebook does.

- **description_exploration.ipynb**: See the 20 attributes that appear the most often per store, to select which ones we keep for cleaning.
- **cleaning_george/**: Folder with notebooks that perform data cleaning for a subset of the stores and save the results to new csv files.
- **cleaning_gerald/**: Folder with notebooks that perform data cleaning for a subset of the stores and save the results to new csv files.
- **data\_cleaning\_andrea**: Notebook that performs data cleaning for a subset of the stores and saves the results to new csv files.
- **clustering\_with\_details.ipynb**: This is our main notebook. Reads the cleaned data, performs some additional cleaning and standarization and calculates the matches. Saves the result to a csv.
- **clustering.ipynb**: First naive approach (and one with highest score), we group by brand, keep only the title and clean it, and find similarities in each cluster (might not work now because data has changed/cleaned).
- **add\_info\_to\_labeled.ipynb**: Add all the information we had to the given labeled dataset for training.
- **ML_approach.ipynb**: Our very unsuccessful machine learning approach.
- **alibaba\_ebay\_clustering.ipynb**: Tried clustering per store but gave up because it was taking too long to run (>12 hours).
- **add_store_cluster_to_brands.ipynb**: Merge the clusters per brand and clusters per store together and save the results to a csv.
- **one\_last\_ride.ipynb**: Our very last effort to try a different approach and increase our score. We put all the cleaned attributes together in a list, group by all the subsets of the list of size two and mark as matches the elements that belong to the same cluster.