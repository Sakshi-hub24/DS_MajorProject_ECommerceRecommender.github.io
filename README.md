# DS_MajorProject_ECommerceRecommender.github.io

# 🛍️ E-Commerce Recommender System
A Data Science project that builds a product recommendation engine based on user behavior such as views, cart additions, and purchases. It also includes an interactive Streamlit web app and Power BI dashboard for data visualization.

---

## 📌 Project Objective
To suggest the top 5 most relevant products to users by analyzing their historical interactions with products using collaborative filtering techniques.

---

## 📁 Project Structure
── DSPR1.py # Model development and recommendation logic
├── StreamlitDashboard.py # Streamlit app to display recommendations
├── ecommerce_recommender_dataset.csv # Cleaned dataset with user behavior
├── user_recommendations.csv # Final user-wise top 5 recommendations
├── user_profiles.csv # (Optional) User details (age, gender)
├── product_info.csv # (Optional) Product details (name, image, rating)
├── PowerBI_Dashboard.pbix # Power BI dashboard file (if included)
└── README.md # Project overview


---

## 🧠 Techniques Used
- **Data Cleaning & Preprocessing** with Pandas
- **User-Item Matrix** creation using pivot tables
- **Collaborative Filtering** with Cosine Similarity
- **Model Evaluation** using RMSE and Precision@K logic
- **Data Visualization** with Power BI
- **Interactive Web App** using Streamlit

---

## 📊 Dashboard Features (Power BI)
- Filters: User ID, Payment Method, Product Category
- KPIs: Total Price, Quantity, Ratings
- Visuals: Pie Chart, Donut Chart, Stacked Bar, Heatmap

---

## 🌐 Streamlit App
- Choose a `user_id` from dropdown
- View their **Top 5 Product Recommendations**
- Built with `StreamlitDashboard.py`
- Simple, clean UI for real-time recommendation display

---

## 🔍 Key Insights
- UPI and Credit Card are top payment methods.
- Users interact with out-of-stock items — showing high demand.
- Metro cities like Delhi and Mumbai show higher buying activity.
- Electronics and Clothing are the most engaged categories.

---

## 🚀 Future Enhancements
- Add **content-based filtering** for cold-start users.
- Show product names, images, and ratings in the app.
- Use **hybrid models** to boost recommendation accuracy.
- Capture and learn from real-time user feedback.

---

## 🛠️ Tools & Libraries
- Python, Pandas, NumPy
- Scikit-learn
- Streamlit
- Power BI
- Matplotlib / Seaborn (for heatmaps)

---

## 👩‍💻 Author
**Sakshi A**  
Tools Used: Python, Streamlit, Power BI  
Feel free to connect and share feedback!
Thank You!

