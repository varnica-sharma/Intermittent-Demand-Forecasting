Intermittent Demand Forecasting for E-commerce Flash Sales
A comprehensive time series analysis project that predicts intermittent demand patterns during e-commerce flash sales using traditional statistical methods, machine learning, and deep learning approaches.

📊 Project Overview
This project implements and compares multiple forecasting methods to predict demand spikes during flash sales—a challenging scenario characterized by:

81.6% zero-demand rate (highly intermittent)

23.6x average demand boost during promotional periods

Long periods of low activity followed by sudden spikes

The framework combines traditional intermittent demand methods with modern deep learning architectures and external factor integration to achieve superior forecasting accuracy.

🎯 Objectives
Predict when and how much demand will occur during flash sales

Compare performance of classical, machine learning, and deep learning approaches

Measure the impact of external factors (promotions, transactions, oil prices, holidays)

Provide practical insights for inventory management and promotion planning

📁 Dataset
Source: Corporación Favorita Grocery Sales Forecasting - Kaggle Competition

The dataset contains historical sales data from Corporación Favorita, a large Ecuadorian-based grocery retailer, including:

train.csv: Daily sales data with store, item, and promotion information

transactions.csv: Daily transaction counts per store

oil.csv: Daily oil price (Ecuador is oil-dependent)

holidays_events.csv: National and regional holidays/events

items.csv: Product metadata (family, class, perishability)

stores.csv: Store metadata (city, state, type, cluster)

Analysis Window

Time Period: August 16, 2016 - August 15, 2017 (365 days)

Products Analyzed: 20 items with high intermittency

Total Records: 7,300 (20 items × 365 days
