import pandas as pd

df = pd.read_csv("DS_MajorPROJ1/ecommerce_recommender_dataset.csv")
print(df)

#check for null values
print(df.isnull().sum())
# Check for duplicate rows
print(df.duplicated().sum())
#describe the dataset
print(df.describe())



# Convert events to interaction scores
event_score = {'view': 1, 'add_to_cart': 2, 'purchase': 3}
df['interaction'] = df['event_type'].map(event_score)


#•	Create a user-item matrix.
user_item_matrix = df.pivot_table(index='user_id', 
                                  columns='product_id', 
                                  values='interaction', 
                                  aggfunc='max', 
                                  fill_value=0)
print(user_item_matrix)
# Visualize the user-item matrix 
import matplotlib.pyplot as plt
# Plot the distribution of interaction scores per event type
interaction_counts = df['event_type'].value_counts()
plt.figure(figsize=(8, 5))
interaction_counts.plot(kind='bar', color=['skyblue', 'orange', 'green'])
plt.title('Distribution of Event Types', fontsize=16, fontweight='bold', color='navy')
plt.xlabel('Event Type', fontsize=12, fontweight='bold')
plt.ylabel('Count', fontsize=12, fontweight='bold')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()




#•	Apply collaborative filtering or matrix factorization.

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
# Compute similarity between users
user_similarity = cosine_similarity(user_item_matrix)
# Convert to DataFrame for readability
user_sim_df = pd.DataFrame(user_similarity, 
                           index=user_item_matrix.index, 
                           columns=user_item_matrix.index)
print(user_sim_df)
# Example for a single user
target_user = user_item_matrix.index[0]
similar_users = user_sim_df[target_user].sort_values(ascending=False)[1:6]
# Products liked by similar users
similar_user_products = user_item_matrix.loc[similar_users.index].sum().sort_values(ascending=False)
# Remove products already seen by target_user
products_seen = user_item_matrix.loc[target_user]
recommended_products = similar_user_products[products_seen == 0].head(5)
print("Top product recommendations for", target_user)
print(recommended_products)


#•	Evaluate with precision k or RMSE.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

#  Train-Test Split
train_df, test_df = train_test_split(df[['user_id', 'product_id', 'interaction']], test_size=0.2, random_state=42)

#  Create user-item interaction matrix from train
train_matrix = train_df.pivot_table(index='user_id', columns='product_id', values='interaction', aggfunc='max', fill_value=0)

#  Compute cosine similarity between users
user_sim = cosine_similarity(train_matrix)
user_sim_df = pd.DataFrame(user_sim, index=train_matrix.index, columns=train_matrix.index)

# Predict interaction using weighted sum of similar users
def predict_interaction(user_id, product_id):
    if user_id not in train_matrix.index or product_id not in train_matrix.columns:
        return 0  # Cold start
    sim_users = user_sim_df[user_id].drop(index=user_id)
    product_values = train_matrix[product_id]

    # Keep only users who interacted with this product
    relevant_users = product_values[product_values > 0].index
    # Only keep relevant_users that are also in sim_users index
    relevant_users_in_sim = [u for u in relevant_users if u in sim_users.index]
    if not relevant_users_in_sim:
        return 0  # No similar users in training set interacted with this product

    sim_users = sim_users[relevant_users_in_sim]

    if sim_users.sum() == 0:
        return 0  # no similar users interacted with this product

    weighted_sum = (sim_users * product_values[relevant_users_in_sim]).sum()
    norm_factor = sim_users.sum()

    return weighted_sum / norm_factor
# Calculate predictions for test data
y_true = []
y_pred = []

for _, row in test_df.iterrows():
    true_val = row['interaction']
    pred_val = predict_interaction(row['user_id'], row['product_id'])

    y_true.append(true_val)
    y_pred.append(pred_val)

#  Compute RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(y_true, y_pred))
print("RMSE:", round(rmse, 4))

#create a heatmap only for numeric columns
import seaborn as sns
plt.figure(figsize=(12, 8))
sns.heatmap(df.select_dtypes(include=['float64', 'int64']).corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Numeric Features', fontsize=16, fontweight='bold', color='navy')
plt.tight_layout()
plt.show()


#3•	Display personalized recommendations.
def get_top_k_recommendations(user_id, k=5):
    if user_id not in user_sim_df.index:
        return []
    similar_users = user_sim_df[user_id].sort_values(ascending=False)[1:6]
    similar_users_data = train_matrix.loc[similar_users.index]
    summed_scores = similar_users_data.sum().sort_values(ascending=False)
    already_seen = train_matrix.loc[user_id][train_matrix.loc[user_id] > 0].index
    recommended = summed_scores.drop(index=already_seen, errors='ignore').head(k).index.tolist()
    return recommended
recommendations = []

for user in train_matrix.index:  # Only users in training data
    top_items = get_top_k_recommendations(user, k=5)
    recommendations.append({
        'user_id': user,
        'recommended_products': top_items
    })

# Convert to DataFrame
rec_df = pd.DataFrame(recommendations)

# Save to CSV
rec_df.to_csv("user_recommendations.csv", index=False)
print("Saved recommendations to 'user_recommendations.csv'")



