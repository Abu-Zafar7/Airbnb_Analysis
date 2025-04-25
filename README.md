# Airbnb Data Analysis Project

## Overview
This project performs an Exploratory Data Analysis (EDA) on the New York City Airbnb dataset from 2019. It provides comprehensive insights into Airbnb listings, including pricing patterns, neighborhood distributions, availability trends, and correlations between various factors.

## Features
- **Data Cleaning and Preprocessing**: Handles missing values, removes outliers, and creates derived features
- **Neighborhood Analysis**: Analyzes listing distribution across NYC neighborhoods
- **Pricing Analysis**: Examines price variations by room type and neighborhood
- **Availability Analysis**: Studies property availability patterns
- **Correlation Analysis**: Identifies relationships between numerical variables
- **Interactive Visualizations**: Creates both static and interactive plots
- **Comprehensive Logging**: Tracks analysis progress and errors

## Project Structure
```
airbnb-analysis/
├── data/               # Data directory
│   └── AB_NYC_2019.csv # Airbnb dataset
├── images/            # Output visualizations
│   ├── *.png          # Static plots
│   └── *.html         # Interactive visualizations
├── scripts/           # Analysis scripts
│   └── eda_airbnb.py  # Main analysis script
├── README.md          # Project documentation
└── requirements.txt   # Python dependencies
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone [your-repository-url]
   cd airbnb-analysis
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**:
   - Place the `AB_NYC_2019.csv` file in the `data/` directory
   - Dataset can be downloaded from [Kaggle](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data)

## Usage

1. **Basic Analysis**:
   ```bash
   python scripts/eda_airbnb.py
   ```

2. **Using Custom Configuration**:
   ```bash
   python scripts/eda_airbnb.py --config path/to/config.json
   ```

## Output Files

### Static Visualizations (PNG)
- `listings_by_neighborhood.png`: Distribution of listings across neighborhoods
- `avg_price_by_roomtype.png`: Average price by room type
- `price_distribution.png`: Price distribution by neighborhood
- `availability_distribution.png`: Distribution of annual availability
- `correlation_matrix.png`: Correlation heatmap of numerical variables

### Interactive Visualizations (HTML)
- `listings_by_neighborhood_interactive.html`: Interactive neighborhood distribution
- `interactive_price_analysis.html`: Interactive price analysis dashboard
- `interactive_availability.html`: Interactive availability analysis
- `interactive_correlation.html`: Interactive correlation matrix

### Logs
- `analysis.log`: Detailed log of the analysis process

## Configuration
The project can be configured using a JSON file with the following parameters:
```json
{
    "data_path": "../data/AB_NYC_2019.csv",
    "images_path": "../images",
    "price_outlier_threshold": 0.99,
    "figure_size": [12, 8],
    "color_palette": "husl",
    "random_state": 42
}
```

## Viewing Interactive Visualizations
1. Navigate to the `images/` directory
2. Open any `.html` file in a web browser
3. Interact with the visualizations:
   - Zoom in/out
   - Hover over data points
   - Pan and rotate
   - Export as PNG if needed

## Dependencies
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- plotly >= 5.3.0
- numpy >= 1.20.0

## Contributing
Feel free to submit issues and enhancement requests!

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Dataset provided by [Kaggle](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data)
- Built with Python data science libraries
