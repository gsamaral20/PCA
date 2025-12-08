# Airline Passenger Satisfaction PCA Analysis

[![NPM](https://img.shields.io/npm/l/react)](https://github.com/devsuperior/sds1-wmazoni/blob/master/LICENSE) 

This project aims to explore and apply Principal Component Analysis (PCA) to a dataset of airline passenger satisfaction. The main goal was to understand the underlying structure of the data by analyzing key PCA outputs such as eigenvalues, factor loadings, communalities, and explained variance.

By applying PCA, the project reduces the dimensionality of the dataset while retaining most of the original information, allowing for a clearer understanding of which variables contribute most to the observed patterns.

## About the Dataset

This dataset contains passenger satisfaction records from **120,000+ airline customers**, including personal characteristics, travel details, and evaluations of the service experience. It is widely used for **classification, clustering, and dimensionality reduction projects**, such as **Principal Component Analysis (PCA)**.

#### Features Description

| Feature | Description |
|--------|-------------|
| **id** | Unique passenger identifier. |
| **Gender** | Passenger gender (Male/Female). |
| **Age** | Passenger age. |
| **Customer Type** | Indicates whether the customer is a *Loyal Customer* or *Disloyal Customer*. |
| **Type of Travel** | Purpose of the trip: *Business* or *Personal* travel. |
| **Class** | Travel class: *Eco*, *Eco Plus*, or *Business*. |
| **Flight Distance** | Distance of the flight in miles. |
| **Departure Delay** | Delay at departure in minutes. |
| **Arrival Delay** | Delay upon arrival in minutes. |
| **Departure and Arrival Time Convenience** | Passenger rating for schedule convenience (1–5). |
| **Ease of Online Booking** | Rating of the online booking process (1–5). |
| **Check-in Service** | Rating of the check-in process (1–5). |
| **Online Boarding** | Rating of the boarding experience (1–5). |
| **Gate Location** | Rating of the gate location convenience (1–5). |
| **On-board Service** | Passenger evaluation of onboard service (1–5). |
| **Seat Comfort** | Seat comfort rating (1–5). |
| **Leg Room Service** | Seating legroom service rating (1–5). |
| **Cleanliness** | Aircraft cleanliness evaluation (1–5). |
| **Food and Drink** | Quality rating of food and beverage services (1–5). |
| **In-flight Service** | General in-flight service rating (1–5). |
| **In-flight Wifi Service** | Rating of Wi-Fi availability/quality during flight (1–5). |
| **In-flight Entertainment** | Entertainment system rating (1–5). |
| **Baggage Handling** | Baggage handling service rating (1–5). |
| **Satisfaction** | Target variable indicating whether the passenger is *Satisfied* or *Neutral/Unsatisfied*. |

The dataset was obtained from Kaggle, and the link can be found [here](https://www.kaggle.com/datasets/mysarahmadbhat/airline-passenger-satisfaction).

## Installation

This project uses **Poetry** for dependency management.  
All dependencies are automatically managed through the `pyproject.toml` file.

To install the environment, run:

poetry install
poetry shell

##  Credits

All the credits to [Gabriel Salles do Amaral](https://www.linkedin.com/in/gabriel-salles-amaral/)