import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import scipy.sparse as sp
import streamlit as st
import warnings

warnings.filterwarnings("ignore")

# Load data
products = pd.read_csv('skincare_products_clean.csv')
chem_df = pd.read_csv('chemicals.csv')

# Remove duplicates
products_df = products.drop_duplicates()
chem_df = chem_df.drop_duplicates()

# Function to find matches
def find_matches(ingredient, chemicals):
    for chemical in chemicals:
        if chemical.lower() in ingredient.lower():
            return chemical
    return None

products_df['matched_chemical'] = products_df['clean_ingreds'].apply(lambda x: find_matches(x, chem_df['Chemical_Name']))

# Merge datasets
skin_products = pd.merge(products_df, chem_df, how='left', left_on='matched_chemical', right_on='Chemical_Name')
skin_products = skin_products.drop(columns=['product_url', 'price', 'Chemical_Name'])
skin_products.dropna(inplace=True)
skin_products = skin_products.drop_duplicates()

grouped_chem = skin_products.groupby('clean_ingreds')['matched_chemical'].apply(lambda x: ', '.join(x)).reset_index()
final_df = skin_products.merge(grouped_chem, on='matched_chemical', how='left')
final_df = final_df[['product_name', 'product_type', 'matched_chemical', 'Skin_Type', 'Description']].drop_duplicates()

# Streamlit app
st.title("Skin Product Recommender")

user_input = st.text_input('Enter your skin type:')

if user_input:
    filtered_products = final_df[final_df['Skin_Type'].str.contains(user_input, case=False, na=False)]
    
    if filtered_products.empty:
        st.write("No products found for the specified skin type.")
    else:
        product_names = filtered_products['product_name'].tolist()
        chemical_ingredients = filtered_products['matched_chemical'].str.strip().str.split(",").tolist()

        def create_bow(chem_list):
            bow = {}
            if not isinstance(chem_list, float):
                for chemical in chem_list:
                    bow[chemical] = 1
            return bow

        bags_of_words = [create_bow(chem_list) for chem_list in chemical_ingredients]
        chem_df_bow = pd.DataFrame(bags_of_words, index=product_names).fillna(0)

        num_features = chem_df_bow.shape[1]
        if num_features < 2:
            st.write("Not enough features for TruncatedSVD. Ensure the input data has sufficient variety.")
        else:
            sparse_chem_df_bow = sp.csr_matrix(chem_df_bow.values)
            n_components = min(10, num_features)
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            reduced_chem_df_bow = svd.fit_transform(sparse_chem_df_bow)

            cosine_sim = cosine_similarity(reduced_chem_df_bow)
            similarity_df = pd.DataFrame(cosine_sim, index=chem_df_bow.index, columns=chem_df_bow.index)

            product = product_names[0]
            product_index = similarity_df.index.get_loc(product)
            top_10 = similarity_df.iloc[product_index].sort_values(ascending=False).iloc[1:11]

            st.write(f'Top 10 similar products to {product}:')
            for item in top_10.index:
                st.write(item)
