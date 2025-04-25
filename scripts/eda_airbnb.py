import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import numpy as np
import logging
import json
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
import sys

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../data/analysis.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "data_path": "../data/AB_NYC_2019.csv",
    "images_path": "../images",
    "price_outlier_threshold": 0.99,
    "figure_size": (12, 8),
    "color_palette": "husl",
    "random_state": 42
}

def load_config(config_path=None):
    """Load configuration from file or use defaults."""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
    else:
        config = DEFAULT_CONFIG
        logger.info("Using default configuration")
    return config

def setup_directories(config):
    """Create necessary directories if they don't exist."""
    directories = [config["data_path"].rsplit('/', 1)[0], config["images_path"]]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")

def load_data(file_path):
    """Load and perform initial data cleaning."""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found at {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return None

def clean_data(df, config):
    """Clean and prepare the dataset."""
    if df is None:
        return None
    
    start_time = time.time()
    
    # Make a copy to avoid modifying original data
    df = df.copy()
    
    # Drop rows with missing names or host names
    df.dropna(subset=["name", "host_name"], inplace=True)
    
    # Fill missing numerical values
    df["reviews_per_month"].fillna(0, inplace=True)
    
    # Convert date
    df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")
    
    # Remove outliers from price
    df = df[df["price"] <= df["price"].quantile(config["price_outlier_threshold"])]
    
    # Add derived features
    df["price_per_night"] = df["price"] / df["minimum_nights"]
    df["reviews_per_month"] = df["number_of_reviews"] / 12  # Approximate
    
    logger.info(f"Data cleaned in {time.time() - start_time:.2f} seconds. New shape: {df.shape}")
    return df

def analyze_neighborhoods(df, config):
    """Analyze and visualize neighborhood data."""
    if df is None:
        return
    
    # Top neighborhoods by listing count
    top_neigh = df["neighbourhood_group"].value_counts()
    logger.info("\nListings by neighborhood group:\n" + str(top_neigh))
    
    # Create visualization
    plt.figure(figsize=config["figure_size"])
    sns.countplot(data=df, x="neighbourhood_group", order=top_neigh.index)
    plt.title("Number of Listings by Neighborhood Group")
    plt.xlabel("Neighborhood Group")
    plt.ylabel("Listing Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{config['images_path']}/listings_by_neighborhood.png")
    plt.close()
    
    # Interactive plot using Plotly
    fig = px.bar(
        x=top_neigh.index,
        y=top_neigh.values,
        title="Number of Listings by Neighborhood Group",
        labels={"x": "Neighborhood Group", "y": "Listing Count"}
    )
    fig.write_html(f"{config['images_path']}/listings_by_neighborhood_interactive.html")

def analyze_pricing(df, config):
    """Analyze and visualize pricing patterns."""
    if df is None:
        return
    
    # Average price by room type
    plt.figure(figsize=config["figure_size"])
    avg_price = df.groupby("room_type")["price"].mean().sort_values()
    sns.barplot(x=avg_price.index, y=avg_price.values)
    plt.title("Average Price by Room Type")
    plt.ylabel("Average Price ($)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{config['images_path']}/avg_price_by_roomtype.png")
    plt.close()
    
    # Price distribution by neighborhood
    plt.figure(figsize=config["figure_size"])
    sns.boxplot(data=df, x="neighbourhood_group", y="price")
    plt.title("Price Distribution by Neighborhood")
    plt.xlabel("Neighborhood Group")
    plt.ylabel("Price ($)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{config['images_path']}/price_distribution.png")
    plt.close()
    
    # Interactive price analysis
    fig = make_subplots(rows=2, cols=1)
    
    # Price by neighborhood
    fig.add_trace(
        go.Box(x=df["neighbourhood_group"], y=df["price"], name="Price by Neighborhood"),
        row=1, col=1
    )
    
    # Price by room type
    fig.add_trace(
        go.Box(x=df["room_type"], y=df["price"], name="Price by Room Type"),
        row=2, col=1
    )
    
    fig.update_layout(height=800, title_text="Interactive Price Analysis")
    fig.write_html(f"{config['images_path']}/interactive_price_analysis.html")

def analyze_availability(df, config):
    """Analyze and visualize availability patterns."""
    if df is None:
        return
    
    # Availability analysis
    plt.figure(figsize=config["figure_size"])
    sns.histplot(data=df, x="availability_365", bins=30)
    plt.title("Distribution of Annual Availability")
    plt.xlabel("Days Available per Year")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{config['images_path']}/availability_distribution.png")
    plt.close()
    
    # Interactive availability analysis
    fig = px.scatter(
        df,
        x="availability_365",
        y="price",
        color="room_type",
        title="Price vs Availability by Room Type",
        labels={"availability_365": "Days Available per Year", "price": "Price ($)"}
    )
    fig.write_html(f"{config['images_path']}/interactive_availability.html")

def analyze_correlations(df, config):
    """Analyze correlations between numerical variables."""
    if df is None:
        return
    
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    # Calculate correlation matrix
    corr_matrix = df[numerical_cols].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=config["figure_size"])
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(f"{config['images_path']}/correlation_matrix.png")
    plt.close()
    
    # Interactive correlation plot
    fig = px.imshow(
        corr_matrix,
        title="Interactive Correlation Matrix",
        color_continuous_scale="RdBu"
    )
    fig.write_html(f"{config['images_path']}/interactive_correlation.html")

def main():
    """Main function to run the analysis."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Airbnb Data Analysis')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    logger.info("Starting Airbnb Data Analysis...")
    start_time = time.time()
    
    # Setup
    setup_directories(config)
    
    # Load and clean data
    df = load_data(config["data_path"])
    df = clean_data(df, config)
    
    if df is not None:
        # Run analyses
        analyze_neighborhoods(df, config)
        analyze_pricing(df, config)
        analyze_availability(df, config)
        analyze_correlations(df, config)
        
        # Print summary statistics
        logger.info("\nSummary Statistics:")
        logger.info("\n" + str(df[["price", "minimum_nights", "availability_365"]].describe()))
        
        logger.info(f"\nAnalysis complete in {time.time() - start_time:.2f} seconds!")
        logger.info(f"Check the {config['images_path']} folder for visualizations.")
    else:
        logger.error("Analysis failed due to data loading issues.")

if __name__ == "__main__":
    main()
