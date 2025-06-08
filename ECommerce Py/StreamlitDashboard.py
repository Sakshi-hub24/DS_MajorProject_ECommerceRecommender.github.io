import streamlit as st
import pandas as pd

# Load dataset
rec_df = pd.read_csv("user_recommendations.csv")
rec_df['recommended_products'] = rec_df['recommended_products'].apply(eval)

# Page setup
st.set_page_config(page_title="🛒 Simple Recommender", layout="centered")

# Styling
st.markdown("""
    <style>
    .stApp {
        background-color: #f4f7fa;
        color: #1a1a1a;
        font-family: 'Segoe UI', sans-serif;
    }

    .title {
        font-size: 30px;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 15px;
    }

    .section {
        font-size: 18px;
        margin-top: 20px;
        color: #34495e;
    }

    .product-card {
        background-color: #ffffff;
        padding: 16px 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 3px 6px rgba(0, 0, 0, 0.05);
        border-left: 5px solid #3498db;
    }

    .footer {
        margin-top: 40px;
        text-align: center;
        font-size: 14px;
        color: #7f8c8d;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='title'>🛍 Product Recommender System</div>", unsafe_allow_html=True)
st.markdown("<div class='section'>Select a user to view their Top 5 recommended products.</div>", unsafe_allow_html=True)

# User selection
selected_user = st.selectbox("Select User", sorted(rec_df['user_id'].unique()))

# Top 5 recommended products
top_products = rec_df[rec_df['user_id'] == selected_user]['recommended_products'].values[0][:5]

st.markdown("### 🎯 Recommended Products:")

# Show products
for i, product_id in enumerate(top_products, 1):
    st.markdown(f"""
        <div class='product-card'>
            <strong>Product {i}</strong><br>
            🆔 Product ID: {product_id}
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("<div class='footer'>📊 Recommender System • Built with Streamlit • By Sakshi</div>", unsafe_allow_html=True)
