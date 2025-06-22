import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from collections import defaultdict
import requests
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from googlesearch import search
import json
import re
import time

# ---------------------- Setup ---------------------- #
google_api_key = "AIzaSyCmMY_SGutER89JBg99YXz3azn4J0F0aYA" 
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.3, api_key=google_api_key)

# Load model and dataset
df = pd.read_csv("dataset_one_hot_encoded.csv")
model = joblib.load("gb_clf_model.pkl")

# Drop cluster columns if they exist
df.drop(columns=[col for col in ["cluster_label", "cluster_category"] if col in df.columns], inplace=True)

# Cluster label to category
cluster_map = {
    0: "Mid-price, moderately volatile, diversified",
    1: "Premium high-volatility stocks",
    2: "High-mid price sector-specific stocks",
    3: "Low-price, stable stocks"
}

# Prompt templates
stock_prompt = ChatPromptTemplate.from_template("""
You are a stock market analyst AI.

Read the following text content extracted from a webpage and extract any stocks or companies recommended for investment TODAY. Provide:
- Stock Name or Ticker
- Reason for recommendation elobarated in detail 
- Confidence score (1 to 10)
- Term: short-term or long-term
- A long paragraph explaining the logic behind recommending these stocks try to sell the stock to the user .

Content:
{content}

Answer in format:
[
  {{ "stock": "XYZ", "reason": "...", "confidence": 8, "term": "short-term", "summary": "..." }},
  ...
]
""")

cluster_prompt = ChatPromptTemplate.from_template("""
You are a stock analyst AI. The model used here is trained on synthetic data, not real-world financial data. Given the following inputs, analyze the predicted cluster category and provide your expert opinion.
before bigining your analysis print top 5 feature value from {mean_features} in an ascii table
include text saying that this is the average daily behaviour of the {company} stock in {year}
Company: {company}
Year: {year}
Mean Feature Stats (Synthetic): 
{mean_features} need top features in table 
montly data {monthly_data} give high weight age and print close prices in table and analyize this in 1 paragraph and why do u think that the stock trends are as it is from your llm knwledge check for news which caused the stock to perofrom in the follwing way {year}
Predicted Cluster: {cluster_label}
Cluster Category: {cluster_category}

Provide 4–5 bullet points with analytical insights or interpretations. Keep it concise and technical do not include any sentence suggesting syntetic data.
""")

# ---------------------- Sidebar Navigation ---------------------- #
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📈 Cluster Predictor", "📰 News-based Stock Picks"])

# ---------------------- Cluster Predictor Page ---------------------- #
if page == "📈 Cluster Predictor":
    st.title("📈 Stock Cluster Predictor")

    companies = sorted(col.replace("Company_", "") for col in df.columns if col.startswith("Company_"))
    selected_company = st.selectbox("Select Company", companies)
    selected_year = st.selectbox("Select Year", sorted(df['Year'].unique()))

    filtered_df = df[(df[f"Company_{selected_company}"] == 1) & (df["Year"] == selected_year)]

    if filtered_df.empty:
        st.warning("No data available for the selected combination.")
    else:
        monthly_data = filtered_df.groupby("Month")["Close"].mean()
        fig, ax = plt.subplots()
        monthly_data.plot(kind='line', marker='o', ax=ax)
        ax.set_title(f"Monthly Close Price for {selected_company} in {selected_year}")
        ax.set_xlabel("Month")
        ax.set_ylabel("Close Price")
        st.pyplot(fig)

        model_features = list(model.feature_names_in_)
        mean_vector = filtered_df[model_features].mean()  
        print(mean_vector)

        for col in model_features:
            if col.startswith("Company_"):
                mean_vector[col] = 1 if col == f"Company_{selected_company}" else 0

        X_input = mean_vector[model_features].values.reshape(1, -1)
        cluster_label = int(model.predict(X_input)[0])
        cluster_category = cluster_map.get(cluster_label, "Unknown")

        st.success(f"Predicted Cluster: {cluster_label}")
        st.info(f"Cluster Category: {cluster_category}")

        with st.spinner("🔍 Analyzing using Gemini..."):
         response = (cluster_prompt | llm).invoke({
        "company": selected_company,
        "year": selected_year,
        "mean_features": mean_vector.to_string(),
        "cluster_label": cluster_label,
        "cluster_category": cluster_category,  # ← ✅ Add comma here
        "monthly_data": monthly_data
    })


        st.markdown("### 🤖 Gemini LLM Recommendation")
        st.write(response.content)

# ---------------------- News-Based Top Stocks Page ---------------------- #
elif page == "📰 News-based Stock Picks":
    st.title("📰 News-based Top Stock Recommendations")

    def get_top_urls(query, num_results=5):
        try:
            return list(search(query, num_results=num_results))
        except Exception as e:
            st.error(f"[ERROR] Google Search failed: {e}")
            return []

    def extract_text_from_url(url):
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            return "\n".join(soup.stripped_strings)[:10000]
        except Exception as e:
            st.error(f"[ERROR] Failed to scrape {url}: {e}")
            return ""

    def analyze_content_with_gemini(content):
        try:
            prompt = stock_prompt.format_messages(content=content)
            return llm.invoke(prompt).content
        except Exception as e:
            st.error(f"[ERROR] Gemini failed: {e}")
            return ""

    def extract_json_block(text):
        try:
            match = re.search(r"\[\s*{.*?}\s*\]", text, re.DOTALL)
            return json.loads(match.group(0)) if match else []
        except Exception as e:
            st.error(f"[ERROR] JSON parsing failed: {e}")
            return []

    def display_stock_recommendations(responses):
        st.markdown("### 📊 Top 5 Stock Picks from Web Content")
        all_stocks = []
        for response in responses:
            parsed = extract_json_block(response)
            for item in parsed:
                if "stock" in item:
                    all_stocks.append(item)

        all_stocks = sorted(all_stocks, key=lambda x: x.get("confidence", 0), reverse=True)[:5]
        if not all_stocks:
            st.warning("No valid recommendations found.")
        else:
            for stock in all_stocks:
                st.markdown(f"**✅ {stock['stock']} ({stock['term']})** → Confidence: {stock['confidence']}/10")
                st.markdown(f"- **Reason**: {stock['reason']}")
                st.markdown(f"- **Summary**: {stock['summary']}\n")

    if st.button("🔎 Get Today's Top 5 Stocks from News"):
        query = "Top 5 stocks to invest today India based on all factors including news"
        urls = get_top_urls(query)

        responses = []
        for i, url in enumerate(urls):
            st.write(f"📄 [{i+1}] Scraping and analyzing: {url}")
            content = extract_text_from_url(url)
            if content:
                result = analyze_content_with_gemini(content)
                responses.append(result)
                time.sleep(1)

        display_stock_recommendations(responses)
