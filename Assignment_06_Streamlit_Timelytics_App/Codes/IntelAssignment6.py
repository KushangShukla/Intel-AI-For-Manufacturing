#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import pandas as pd

os.chdir("C:\\Users\\kusha\\OneDrive\\Desktop\\Kushang's Files\\Intel AI Course\\Week 6 - Dependencies\\Dependencies\\data")
base_dir = os.getcwd()  

print(f"Current working directory: {base_dir}")

# File names
csv_files = {
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "product_category_translation": "product_category_name_translation.csv"
}

# Load CSVs into DataFrames
dataframes = {}

for key, file in csv_files.items():
    try:
        df = pd.read_csv(file)
        dataframes[key] = df
        print(f"✅ Loaded: {file} — Shape: {df.shape}")
    except FileNotFoundError:
        print(f"❌ File not found: {file}")

# Example: View top rows from each
for name, df in dataframes.items():
    print(f"\n🔹 Preview: {name} — {df.shape}")
    print(df.head(2))


# In[5]:


orders = pd.read_csv("olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp", "order_delivered_customer_date"])


# Convert dates
orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
orders["order_delivered_customer_date"] = pd.to_datetime(orders["order_delivered_customer_date"])

# Drop missing values for delivered orders
orders = orders.dropna(subset=["order_purchase_timestamp", "order_delivered_customer_date"])

# Create a new column for delivery time in days
orders["delivery_time_days"] = (orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"]).dt.days


# In[6]:


order_items = pd.read_csv("olist_order_items_dataset.csv")
products = pd.read_csv("olist_products_dataset.csv")
customers = pd.read_csv("olist_customers_dataset.csv")


# Merge with order_items to get product_id
orders_merged = pd.merge(orders, order_items, on="order_id", how="left")

# Merge with products to get product_category
orders_merged = pd.merge(orders_merged, products[["product_id", "product_category_name"]], on="product_id", how="left")

# Merge with customers to get customer state
orders_merged = pd.merge(orders_merged, customers[["customer_id", "customer_state"]], on="customer_id", how="left")

# Clean nulls
orders_merged = orders_merged.dropna(subset=["delivery_time_days", "product_category_name", "customer_state"])


# In[7]:


# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import os

# Change directory and load data
os.chdir("C:\\Users\\kusha\\OneDrive\\Desktop\\Kushang's Files\\Intel AI Course\\Week 6 - Dependencies\\Dependencies\\data")

# Load relevant datasets only
orders = pd.read_csv("olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp", "order_delivered_customer_date"])
order_items = pd.read_csv("olist_order_items_dataset.csv")
products = pd.read_csv("olist_products_dataset.csv")
customers = pd.read_csv("olist_customers_dataset.csv")

# Preprocessing
orders = orders.dropna(subset=["order_purchase_timestamp", "order_delivered_customer_date"])
orders["delivery_time_days"] = (orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"]).dt.days

# Merging for rich info
df = orders.merge(order_items, on="order_id")
df = df.merge(products[["product_id", "product_category_name"]], on="product_id")
df = df.merge(customers[["customer_id", "customer_state"]], on="customer_id")
df = df.dropna(subset=["delivery_time_days", "product_category_name", "customer_state"])

# UI
st.set_page_config("Timelytics - Delivery Time Predictor")
st.title("📦 Timelytics - Delivery Time Prediction App")
st.markdown("Enter order details to estimate delivery time based on historical Olist data.")

# Inputs
category = st.selectbox("Product Category", sorted(df["product_category_name"].unique()))
state = st.selectbox("Customer State", sorted(df["customer_state"].unique()))

if st.button("Predict Delivery Time"):
    avg_days = df[
        (df["product_category_name"] == category) &
        (df["customer_state"] == state)
    ]["delivery_time_days"].mean()

    if np.isnan(avg_days):
        st.warning("⚠️ Not enough data for this combination.")
    else:
        st.success(f"🕒 Estimated Delivery Time: **{round(avg_days, 2)} days**")

# Extra: Chart
st.subheader("📊 Average Delivery Time by Shipping State")
avg_by_state = df.groupby("customer_state")["delivery_time_days"].mean().sort_values()
st.bar_chart(avg_by_state)

