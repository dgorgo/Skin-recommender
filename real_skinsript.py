# %%
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import scipy.sparse as sp
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer



# %%
products = pd.read_csv('skincare_products_clean.csv')
chem_df = pd.read_csv('chemicals.csv')

# %%
products

# %%
chem_df

# %%
products.size

# %%
chem_df.size

# %% Remove duplicates
products_df = products.drop_duplicates()
chem_df = chem_df.drop_duplicates()

# %%
products_df.size

# %%
chem_df.size

# %% Function to find matches
def find_matches(ingredient, chemicals):
    for chemical in chemicals:
        if chemical.lower() in ingredient.lower():
            return chemical
    return None

products_df['matched_chemical'] = products_df['clean_ingreds'].apply(lambda x: find_matches(x, chem_df['Chemical_Name']))

# %% Merge datasets
skin_products = pd.merge(products_df, chem_df, how='left', left_on='matched_chemical', right_on='Chemical_Name')
skin_products = skin_products.drop(columns=['product_url', 'price', 'Chemical_Name'])
skin_products.dropna(inplace=True)

# %% Remove duplicates after merging
skin_products = skin_products.drop_duplicates()

# %%

grouped_chem = skin_products.groupby('clean_ingreds')['matched_chemical'].apply(lambda x: ', '.join(x)).reset_index()
final_df = skin_products.merge(grouped_chem, on='matched_chemical', how='left')
final_df = final_df[['product_name', 'product_type', 'matched_chemical', 'Skin_Type', 'Description']]


# %%
final_df = final_df.drop_duplicates()

# Streamlit app layout
st.title("Skin Care Product Recommendation System")

# Get user input for skin type
user_input = st.selectbox('What is your skin type?', ['Dry Skin', 'Oily Skin', 'Combination Skin', 'Normal Skin'])
#user_input = st.text_input('What is your skin type:')


# %% Filter products based on the input skin type
if user_input:
    
   filtered_products = final_df[final_df['Skin_Type'].str.contains(user_input, case=False, na=False)]
   if filtered_products.empty:
       
      st.write("No products found for the specified skin type.")
   else:
      product_names = filtered_products['product_name'].tolist()
      chemical_ingredients = filtered_products['matched_chemical'].str.strip().str.split(",").tolist()
       

    # Create bags of words (bow)
    
       def create_bow(chem_list):
           bow = {}
           if not isinstance(chem_list, float):
               for chemical in chem_list:
                   bow[chemical] = 1
            return bow

    bags_of_words = [create_bow(chem_list) for chem_list in chemical_ingredients]
    chem_df_bow = pd.DataFrame(bags_of_words, index=product_names).fillna(0)


# %% Check dimensions of chem_df_bow
num_features = chem_df_bow.shape[1]
if num_features < 2:
        print("Not enough features for TruncatedSVD. Ensure the input data has sufficient variety.")
else:
        #Transform Data to Sparse Matrix Format and Apply Dimensionality Reduction
    sparse_chem_df_bow = sp.csr_matrix(chem_df_bow.values)

# %% Print the number of features
num_features = chem_df_bow.shape[1]
print(f'Number of features: {num_features}')

# %% Use TruncatedSVD to reduce dimensionality
n_components = min(10, num_features)
svd = TruncatedSVD(n_components=n_components, random_state=42)
reduced_chem_df_bow = svd.fit_transform(sparse_chem_df_bow)

# %% Calculate cosine similarity on the reduced data
cosine_sim = cosine_similarity(reduced_chem_df_bow)
similarity_df = pd.DataFrame(cosine_sim, index=chem_df_bow.index, columns=chem_df_bow.index)

# %%
# Get recommendations for a specific product in the filtered set
product = st.selectbox('Select a product for recommendations:', product_names)
if product:
            product_index = similarity_df.index.get_loc(product)
            top_10 = similarity_df.iloc[product_index].sort_values(ascending=False).iloc[1:11]

            # Print the top 10 most similar products
            st.write(f'Top 10 similar products to {product}:')
            for similar_product in top_10.index:
                st.write(similar_product)
# %%

        #Get recommendations for a specific product in the filtered set
#product = product_names[0]  # Select the first product as a reference
#product_index = similarity_df.index.get_loc(product)
#top_10 = similarity_df.iloc[product_index].sort_values(ascending=False).iloc[1:11]

# %% Print the top 10 most similar products
#print(f'Top 10 similar products to {product}:')
#print(top_10)

# %%


# %%


# %%



