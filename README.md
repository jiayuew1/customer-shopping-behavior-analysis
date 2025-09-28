# Customer Shopping Behavior Analysis

## Project Overview
Comprehensive analysis of 3,900 customer shopping records to identify high-value customer segments, purchasing patterns, and strategic business opportunities. This project demonstrates advanced data analysis, customer segmentation, and business intelligence techniques.

## Dataset
- **Size**: 3,900 customer records with 18 attributes
- **Source**: Customer shopping behavior data
- **Time Period**: Multi-seasonal retail data
- **Data Quality**: Complete dataset with no missing values

## Key Business Findings

### Customer Segmentation
- **4-Tier Value System**: Bronze, Silver, Gold, Platinum customers
- **Platinum Customers**: 25% of customers generate 28.1% of total revenue ($65,468)
- **High-Value Profile**: 69% male, avg age 44.5, prefer clothing category

### Market Insights
- **Top Category**: Clothing (44.5% of orders, $104,264 revenue)
- **Peak Season**: Fall generates highest average order value ($61.56)
- **Customer Lifetime**: Average 25.3 previous purchases
- **Satisfaction**: 3.75/5.0 average rating across all segments

### Behavioral Drivers (PCA Analysis)
- **PC1 (21.7% variance)**: Marketing engagement (discounts, promos, subscriptions)
- **PC2 (11.6% variance)**: Product category preferences
- **Key Finding**: Marketing tools and product categories drive 33.4% of customer behavior

## Technical Implementation

### Technologies Used
- **Python 3.x**: Core analysis platform
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: PCA and machine learning
- **Matplotlib/Seaborn**: Statistical visualizations
- **Plotly**: Interactive geographic mapping

### Analysis Modules
1. **Data Quality Assessment**: Comprehensive data profiling
2. **Demographics Analysis**: Customer base characteristics with US geographic mapping
3. **Product Performance**: Category and seasonal analysis
4. **Customer Segmentation**: Value-based tier identification
5. **PCA Factor Analysis**: Behavioral driver identification
6. **Strategic Recommendations**: Actionable business insights

## Key Metrics
- **Total Revenue**: $233,081
- **Average Order Value**: $59.76
- **Customer Retention**: 27% subscription rate
- **Marketing Penetration**: 43% discount usage
- **Geographic Reach**: 50 US states

## Strategic Recommendations

### Customer Strategy
- **Target Platinum Segment**: Focus acquisition on male customers 40-50 with clothing preferences
- **Frequency Optimization**: Convert quarterly buyers to monthly (13.8% currently weekly)
- **Subscription Growth**: Increase from current 27% to 35% target

### Product Strategy
- **Clothing Focus**: Expand inventory in highest-performing category
- **Seasonal Planning**: Optimize fall inventory for peak performance
- **Cross-selling**: Leverage accessories category for upselling

### Marketing Optimization
- **Discount Strategy**: Current 43% usage shows strong engagement
- **Segmented Campaigns**: Develop tier-specific marketing approaches
- **Quality Improvement**: Address 3.75/5.0 rating through product enhancement

## Repository Structure
```
├── src/customer_analyzer.py    # Main analysis script
├── data/shopping_data.csv      # Dataset
├── results/                    # Analysis outputs and visualizations
├── docs/                       # Documentation and reports
└── requirements.txt            # Dependencies
```

## Installation & Usage

```bash
# Clone repository
git clone https://github.com/yourusername/customer-shopping-behavior-analysis.git

# Install dependencies
pip install -r requirements.txt

# Run analysis
python src/customer_analyzer.py
```

## Business Impact
This analysis provides actionable insights for:
- **Revenue Growth**: Target high-value customer acquisition
- **Marketing ROI**: Optimize promotional strategies
- **Inventory Management**: Data-driven seasonal planning
- **Customer Retention**: Develop segment-specific loyalty programs

## Future Enhancements
- Real-time dashboard implementation
- Predictive modeling for customer churn
- Advanced recommendation engine
- A/B testing framework for marketing strategies

## Author
Jiayue Wang 

## License
This project is licensed under the MIT License - see the LICENSE file for details.
