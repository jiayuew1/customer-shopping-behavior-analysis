
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import seaborn as sns
import warnings
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_palette("husl")
class CustomerShoppingAnalyzer:
    """
    Customer Shopping Behavior & Product Performance Analysis
    Dataset: 3,900 customer records with 18 attributes
    Focus: Product Performance, Customer Segmentation, Sales Analytics
    """
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        
    def load_and_explore_data(self):
        """
        Load the customer shopping data and perform initial exploration
        """
        print("CUSTOMER SHOPPING BEHAVIOR & PRODUCT PERFORMANCE ANALYSIS")
        print("=" * 70)
        print("Dataset: 3,900 customer records with 18 detailed attributes")
        print("Objective: Extract Business Insights for Product & Marketing Strategy")
        print("=" * 70)
        try:
            # Load the data
            import os
            files = os.listdir(self.data_path)
            target_file = 'shopping_behavior_updated.csv'        
            loaded = False
            try:
                self.df = pd.read_csv(os.path.join(self.data_path, target_file))
                loaded = True
            except Exception as e:
                print(f"Failed to load {target_file}: {e}")
            
            if not loaded:
                # Load CSV file safely
                csv_files = [f for f in files if f.endswith('.csv')]
                if csv_files:
                    filename = csv_files[0]
                    print(f"Trying {filename}...")
                    self.df = pd.read_csv(os.path.join(self.data_path, filename))
                    print(f"Loaded {filename}")
                    loaded = True
            
            if not loaded:
                print("Could not find suitable CSV file")
                return False
                
            print(f"Dataset shape: {self.df.shape}")
            print(f"Columns: {list(self.df.columns)}")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    def data_quality_overview(self):
        """
        Data quality check and overview analysis
        """
        print("\n" + "="*60)
        print("DATA QUALITY & BUSINESS OVERVIEW")
        print("="*60)
        
        # Basic dataset info
        print("DATASET SUMMARY:")
        print(f"Total Records: {len(self.df):,}")
        print(f"Total Columns: {len(self.df.columns)}")
        print(f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Display sample data
        print(f"\nSAMPLE DATA:")
        print(self.df.head())
        
        # Data types
        print(f"\nDATA TYPES:")
        print(self.df.dtypes)
        
        # Missing values analysis
        print(f"\nMISSING VALUES ANALYSIS:")
        missing_data = self.df.isnull().sum()
        missing_pct = (missing_data / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing_data,
            'Missing_Percentage': missing_pct
        }).sort_values('Missing_Count', ascending=False)
        
        if missing_data.sum() == 0:
            print("No missing values found")
        else:
            print(missing_df[missing_df.Missing_Count > 0])
        
        # Basic statistics for numerical columns
        print(f"\nNUMERICAL COLUMNS OVERVIEW:")
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            print(self.df[numerical_cols].describe().round(2))
        
        # Categorical columns overview
        print(f"\nCATEGORICAL COLUMNS OVERVIEW:")
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            unique_count = self.df[col].nunique()
            print(f"   {col}: {unique_count} unique values")
            if unique_count <= 10:  # Show values for small categories
                print(f"      Values: {list(self.df[col].unique())}")

    def customer_demographics_analysis(self):
        """
        Analyze customer demographics
        """
        print("\n" + "="*60)
        print("CUSTOMER DEMOGRAPHICS ANALYSIS")
        print("="*60)

        fig, axes = plt.subplots(2, 3, figsize=(20,12))
        fig.suptitle('Customer Demographics & Behavior Overview', fontsize=16, fontweight='bold', y=0.95)
        # 1. Age Groups (More intuitive than histogram)
        if 'Age' in self.df.columns:
            # Create meaningful age groups
            age_bins = [0, 25, 35, 45, 55, 100]
            age_labels = ['18-25', '26-35', '36-45', '46-55', '55+']
            self.df['Age_Group'] = pd.cut(self.df['Age'], bins=age_bins, labels=age_labels, include_lowest=True)
            
            age_counts = self.df['Age_Group'].value_counts().sort_index()
            bars = axes[0,0].bar(range(len(age_counts)), age_counts.values, 
                                color='steelblue', alpha=0.8, edgecolor='white', linewidth=1)
            
            axes[0,0].set_title('Customer Age Groups', fontsize=14, fontweight='bold', pad=20)
            axes[0,0].set_xlabel('Age Group', fontsize=12)
            axes[0,0].set_ylabel('Number of Customers', fontsize=12)
            axes[0,0].set_xticks(range(len(age_counts)))
            axes[0,0].set_xticklabels(age_counts.index, fontsize=11)

            axes[0,0].grid(True, alpha=0.3, axis='y')
            axes[0,0].set_ylim(0, max(age_counts.values) * 1.2)
        
        # 2. Gender with better colors and labels
        if 'Gender' in self.df.columns:
            gender_counts = self.df['Gender'].value_counts()
            colors = ['#FF6B9D', '#4ECDC4']  # More appealing colors
            
            wedges, texts, autotexts = axes[0,1].pie(
                gender_counts.values, 
                labels=[f'{gender}\n({count:,} customers)' for gender, count in gender_counts.items()],
                autopct='%1.1f%%',
                startangle=90,
                colors=colors,
                textprops={'fontsize': 12, 'fontweight': 'bold'},
                explode=(0.05, 0.05)  # Slight separation
            )
            
            axes[0,1].set_title('Gender Distribution', fontsize=14, fontweight='bold', pad=20)
            
        
        # 3. Purchase Amount Ranges (More meaningful than raw histogram)
        if 'Purchase Amount (USD)' in self.df.columns:
            # Create spending tiers
            axes[0,2].hist(self.df['Purchase Amount (USD)'], bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
            axes[0,2].set_title('Purchase Amount', fontsize=14, pad=15,  fontweight='bold')
            axes[0,2].set_xlabel('Amount ($)')
            axes[0,2].set_ylabel('Frequency')
            axes[0,2].grid(True, alpha=0.3, axis='y')
            
            # Add average line
            avg_amount = self.df['Purchase Amount (USD)'].mean()
            axes[0,2].axvline(avg_amount, color='red', linestyle='--', 
                            label=f'Avg: ${avg_amount:.0f}')
            axes[0,2].legend(fontsize=9)
        # 4. Purchase Frequency Distribution (ordered by actual frequency)
        if 'Frequency of Purchases' in self.df.columns:
            frequency_counts = self.df['Frequency of Purchases'].value_counts()
            
            # Order by actual purchase frequency (most frequent to least frequent)
            frequency_order = ['Weekly', 'Bi-Weekly', 'Fortnightly', 'Monthly', 
                            'Quarterly', 'Every 3 Months', 'Annually']
            
            # Reorder frequency counts based on logical order
            frequency_ordered = frequency_counts.reindex(frequency_order, fill_value=0)
            
            # Create gradient colors from red (high) to blue (low)
            n_bars = len(frequency_ordered)
            colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, n_bars))
            
            bars = axes[1,0].bar(range(len(frequency_ordered)), frequency_ordered.values, 
                                color=colors, alpha=0.8, edgecolor='white', linewidth=1)
            
            axes[1,0].set_title('Customer Purchase Frequency Distribution', fontsize=14,fontweight='bold', pad=15)
            axes[1,0].set_ylabel('Number of Customers')
            axes[1,0].set_xticks(range(len(frequency_ordered)))
            axes[1,0].set_xticklabels(frequency_ordered.index, fontsize=9, ha='right', rotation=45)
            axes[1,0].grid(True, alpha=0.3, axis='y')
            
        # 5. Category Performance with revenue
        if 'Category' in self.df.columns:
            category_stats = self.df.groupby('Category').agg({
                'Customer ID': 'count',
                'Purchase Amount (USD)': ['sum', 'mean'] if 'Purchase Amount (USD)' else lambda x: None
            })
            
            if 'Purchase Amount (USD)':
                category_stats.columns = ['Order_Count', 'Total_Revenue', 'Avg_Order_Value']
                category_stats = category_stats.sort_values('Total_Revenue', ascending=False)
            else:
                category_stats.columns = ['Order_Count']
                category_stats = category_stats.sort_values('Order_Count', ascending=False)
            
            bars = axes[1,1].bar(range(len(category_stats)), category_stats['Order_Count'].values,
                                color='mediumpurple', alpha=0.8, edgecolor='white', linewidth=1)
            
            axes[1,1].set_title('Product Category Performance', fontsize=14, fontweight='bold', pad=20)
            axes[1,1].set_xlabel('Category', fontsize=12)
            axes[1,1].set_ylabel('Number of Orders', fontsize=12)
            axes[1,1].set_xticks(range(len(category_stats)))
            axes[1,1].set_xticklabels(category_stats.index, fontsize=10, rotation=45, ha='right') 
            axes[1,1].grid(True, alpha=0.3, axis='y')
            max_amount = category_stats['Order_Count'].max()
            axes[1,1].set_ylim(0, max_amount * 1.2)
        
        # 6. Review Ratings with satisfaction levels
        if 'Review Rating' in self.df.columns:
            # Group ratings into satisfaction levels
            def rating_category(rating):
                if rating <= 2:
                    return 'Poor (1-2)'
                elif rating <= 3:
                    return 'Average (3)'
                elif rating <= 4:
                    return 'Good (4)'
                else:
                    return 'Excellent (5)'
            
            self.df['Satisfaction_Level'] = self.df['Review Rating'].apply(rating_category)
            satisfaction_counts = self.df['Satisfaction_Level'].value_counts()
            
            # Order by satisfaction level
            order = ['Poor (1-2)', 'Average (3)', 'Good (4)', 'Excellent (5)']
            satisfaction_counts = satisfaction_counts.reindex([x for x in order if x in satisfaction_counts.index])
            
            colors_satisfaction = ['#FF4444', '#FFAA44', '#44AAFF', '#44FF44']
            bars = axes[1,2].bar(range(len(satisfaction_counts)), satisfaction_counts.values,
                                color=colors_satisfaction[:len(satisfaction_counts)], 
                                alpha=0.8, edgecolor='white', linewidth=1)
            
            axes[1,2].set_title('Customer Satisfaction Levels', fontsize=14, fontweight='bold', pad=20)
            axes[1,2].set_xlabel('Satisfaction Level', fontsize=12)
            axes[1,2].set_ylabel('Number of Reviews', fontsize=12)
            axes[1,2].set_xticks(range(len(satisfaction_counts)))
            axes[1,2].set_xticklabels(satisfaction_counts.index, fontsize=10, rotation=45, ha='right')
            axes[1,2].grid(True, alpha=0.3, axis='y')
            axes[1,2].set_ylim(0, max(satisfaction_counts.values) * 1.2)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(left=0.05, top=0.78, bottom = 0.15, hspace=0.5)
        plt.show()

        print("KEY INSIGHTS:")

        dominant_age = self.df['Age_Group'].value_counts().index[0]
        print(f"  Dominant Age Group: {dominant_age}")
        gender_dist = self.df['Gender'].value_counts(normalize=True) * 100
        majority_gender = gender_dist.index[0]
        print(f"  Gender Split: {majority_gender} {gender_dist.iloc[0]:.1f}%")
        print(f"  Average Purchase: ${self.df['Purchase Amount (USD)'].mean():.2f}")
        print(f"  Purchase Range: ${self.df['Purchase Amount (USD)'].min():.0f} - ${self.df['Purchase Amount (USD)'].max():.0f}")
        avg_rating = self.df['Review Rating'].mean()
        print(f"  Average Rating out of 5: {avg_rating:.2f}")

    def product_performance_analysis(self):
        """
        Detailed product performance and category analysis
        """
        print("\n" + "="*60)
        print("PRODUCT PERFORMANCE ANALYSIS")
        print("="*60)
        
        # 1. Category Performance
        print("CATEGORY PERFORMANCE METRICS:")
        category_metrics = self.df.groupby('Category').agg({
            'Purchase Amount (USD)': ['count', 'sum', 'mean'],
            'Review Rating': 'mean'
        }).round(2)
        
        category_metrics.columns = ['Order_Count', 'Total_Revenue', 'Avg_Order_Value', 'Avg_Rating']
        category_metrics = category_metrics.sort_values('Total_Revenue', ascending=False)
        print(category_metrics)
        
        # 2. Seasonal Performance
        print(f"\nSEASONAL PERFORMANCE ANALYSIS:")
        seasonal_metrics = self.df.groupby('Season').agg({
            'Purchase Amount (USD)': ['count', 'sum', 'mean']
        }).round(2)
        seasonal_metrics.columns = ['Order_Count', 'Total_Revenue', 'Avg_Order_Value']
        seasonal_metrics = seasonal_metrics.sort_values('Total_Revenue', ascending=False)
        print(seasonal_metrics)
        
        # 3. Size and Color Analysis
        print(f"\nSIZE POPULARITY:")
        size_dist = self.df['Size'].value_counts()
        print(size_dist)
        
        print(f"\nCOLOR PREFERENCES:")
        color_dist = self.df['Color'].value_counts().head(10)
        print(color_dist)
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Product Performance Analysis', fontsize=16, fontweight='bold')
        
        # Category revenue
        category_metrics['Total_Revenue'].plot(kind='bar', ax=axes[0,0], color='green')
        axes[0,0].set_title('Total Revenue by Category')
        axes[0,0].set_ylabel('Revenue (USD)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Seasonal trends
        seasonal_metrics['Total_Revenue'].plot(kind='bar', ax=axes[0,1], color='orange')
        axes[0,1].set_title('Revenue by Season')
        axes[0,1].set_ylabel('Revenue (USD)')
        axes[0,1].tick_params(axis='x', rotation=0)
        
        # Size distribution
        size_dist.plot(kind='bar', ax=axes[1,0], color='purple')
        axes[1,0].set_title('Size Distribution')
        axes[1,0].set_ylabel('Count')
        axes[1,0].tick_params(axis='x', rotation=0)
        
        # Color preferences
        color_dist.plot(kind='bar', ax=axes[1,1], color='coral')
        axes[1,1].set_title('Top 10 Color Preferences')
        axes[1,1].set_ylabel('Count')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    def create_us_map_visualization(self):
        """
        Create complete US map with all 50 states - fixed version
        """
        if 'Location' not in self.df.columns:
            print("No Location column found")
            return
        
        location_counts = self.df['Location'].value_counts()
        
        # Complete state abbreviations mapping
        state_abbr_map = {
            'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
            'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
            'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
            'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
            'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
            'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
            'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
            'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
            'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
            'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
        }
        
        # Complete US state positions (approximated grid layout)
        state_positions = {
            # Northern states, including the Upper Midwest
            'Washington': (0, 0), 'Idaho': (0, 1), 'Montana': (0, 2), 'North Dakota': (0, 3), 'Minnesota': (0, 4), 
            'Wisconsin': (0, 5), 'Michigan': (0, 6), 'New York': (0, 7), 'Vermont': (0, 8), 'New Hampshire': (0, 9), 
            'Maine': (0, 10),
            
            # Central-North states
            'Oregon': (1, 0), 'Wyoming': (1, 2), 'South Dakota': (1, 3), 'Iowa': (1, 4), 'Illinois': (1, 5), 
            'Indiana': (1, 6), 'Ohio': (1, 7), 'Pennsylvania': (1, 8), 'New Jersey': (1, 9), 'Connecticut': (1, 10), 
            'Rhode Island': (1, 11), 'Massachusetts': (1, 12),
            
            # Mid-states
            'California': (2, 0), 'Nevada': (2, 1), 'Utah': (2, 2), 'Colorado': (2, 3), 'Nebraska': (2, 4), 
            'Missouri': (2, 5), 'Kentucky': (2, 6), 'West Virginia': (2, 7), 'Virginia': (2, 8), 'Maryland': (2, 9), 
            'Delaware': (2, 10),
            
            # Southern states
            'Arizona': (3, 2), 'New Mexico': (3, 3), 'Kansas': (3, 4), 'Arkansas': (3, 5), 'Tennessee': (3, 6), 
            'North Carolina': (3, 7), 'South Carolina': (3, 8),
            
            # Deep South and Texas
            'Oklahoma': (4, 4), 'Texas': (4, 3), 'Louisiana': (4, 5), 'Mississippi': (4, 6), 'Alabama': (4, 7), 
            'Georgia': (4, 8), 'Florida': (4, 9),
            
            # Alaska and Hawaii (separate positions)
            'Alaska': (5, 0), 'Hawaii': (5, 1)
        }
            
        # Create figure
        fig, ax = plt.subplots(figsize=(22, 14))
        
        # Get color mapping
        if len(location_counts) > 0:
            min_count = location_counts.min()
            max_count = location_counts.max()
            
            def get_color(count):
                if count == 0:
                    return 'lightgray'
                if max_count == min_count:
                    return 'steelblue'
                # Normalize count to 0-1 range
                normalized = (count - min_count) / (max_count - min_count)
                return plt.cm.Blues(0.3 + normalized * 0.7)
        else:
            def get_color(count):
                return 'lightgray'
        
        # Draw all 50 states (whether they have data or not)
        for state, (row, col) in state_positions.items():
            # Get customer count for this state
            count = location_counts.get(state, 0)
            
            # Get state abbreviation
            abbr = state_abbr_map[state]
            
            # Determine color and transparency
            if count > 0:
                color = get_color(count)
                alpha = 0.8
                text_color = 'white' if count > max_count * 0.5 else 'black'
            else:
                color = 'lightgray'
                alpha = 0.4
                text_color = 'darkgray'
            
            # Create rectangle - make them touching by reducing spacing
            rect = plt.Rectangle((col * 1.0, -row * 1.0), 0.95, 0.95, 
                            facecolor=color, 
                            edgecolor='white', 
                            linewidth=1,
                            alpha=alpha)
            ax.add_patch(rect)
            
            # Add state abbreviation (for ALL states)
            ax.text(col * 1.0 + 0.475, -row * 1.0 + 0.65, abbr, 
                ha='center', va='center', 
                fontsize=8, fontweight='bold',
                color=text_color)
            
            # Add customer count (only if > 0)
            if count > 0:
                ax.text(col * 1.0 + 0.475, -row * 1.0 + 0.25, f'{count}', 
                    ha='center', va='center', 
                    fontsize=7, fontweight='bold',
                    color='darkblue')
            
        # Calculate map dimensions for centering
        max_col = max(col for _, (row, col) in state_positions.items())
        max_row = max(row for _, (row, col) in state_positions.items())
        
        # Corrected xlim to center the map
        ax.set_xlim(-1, max_col + 1)
        
        # Corrected ylim to add vertical padding
        ax.set_ylim(-max_row - 1.6, 1.6)
        
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add title
        plt.title('Customer Distribution Across All US States',
                fontsize=18, fontweight='bold', y=0.95)
        
        # Create comprehensive legend
        from matplotlib.patches import Patch
        legend_elements = []
        
        if len(location_counts) > 0:
            # Top performer
            top_state = location_counts.index[0]
            top_count = location_counts.iloc[0]
            legend_elements.append(Patch(color=get_color(top_count), 
                                    label=f'Highest: {top_state} ({top_count} customers)'))
            
            # Color scale legend
            if len(location_counts) > 1:
                legend_elements.append(Patch(color=plt.cm.Blues(0.8), 
                                        label=f'High volume: {max_count//2}+ customers'))
                legend_elements.append(Patch(color=plt.cm.Blues(0.5), 
                                        label=f'Medium volume: 1-{max_count//2} customers'))
            
            legend_elements.append(Patch(color='lightgray', alpha=0.4,
                                    label='No customers'))
            
            ax.legend(handles=legend_elements, loc='upper left',
                    bbox_to_anchor=(0.8, 0.42), fontsize=11)
        
        # Add summary statistics
        total_states_with_customers = len(location_counts)
        market_penetration = (total_states_with_customers / 50) * 100
        
        stats_text = f"""Market Coverage Summary:
    - States with customers: {total_states_with_customers}/50
    - Market penetration: {market_penetration:.1f}%
    - Total customers: {len(self.df):,}
    - Average per active state: {len(self.df) / total_states_with_customers:.1f}
    - Top market: {location_counts.index[0]} ({location_counts.iloc[0]} customers)"""
        
        ax.text(0.8, 0.1, stats_text, transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
            verticalalignment='bottom')
        
        plt.tight_layout()
        plt.show()
        
        return location_counts
    # Add this method to your CustomerShoppingAnalyzer class
    print("Add create_complete_us_map(self) method to your CustomerShoppingAnalyzer class")
    print("Then call: analyzer.create_complete_us_map()")
    
    def customer_segmentation_analysis(self):
        """
        High-Value Customer Identification & Segmentation
        Focus: Identify and profile most valuable customers
        Goal: Customer prioritization and targeted strategies
        """
        print("\n" + "="*60)
        print("HIGH-VALUE CUSTOMER IDENTIFICATION")
        print("="*60)
        
        # 1. Customer Value Scoring (Purchase Amount + Previous Purchases)
        print("CUSTOMER VALUE SEGMENTATION:")
        
        # Create value score combining current purchase and loyalty
        self.df['Value_Score'] = (
            self.df['Purchase Amount (USD)'] * 0.6 +  # Current purchase weight
            self.df['Previous Purchases'] * 10 * 0.4   # Loyalty weight (scaled up)
        )
        
        # Segment customers into value tiers
        self.df['Value_Tier'] = pd.qcut(self.df['Value_Score'], 
                                       q=4, 
                                       labels=['Bronze', 'Silver', 'Gold', 'Platinum'])
        
        # Analyze each tier
        tier_analysis = self.df.groupby('Value_Tier').agg({
            'Purchase Amount (USD)': ['count', 'mean', 'sum'],
            'Previous Purchases': 'mean',
            'Review Rating': 'mean',
            'Age': 'mean',
            'Subscription Status': lambda x: (x == 'Yes').mean() * 100,
            'Value_Score': 'mean'
        }).round(2)
        
        tier_analysis.columns = ['Customer_Count', 'Avg_Purchase', 'Total_Revenue', 
                               'Avg_Previous_Purchases', 'Avg_Rating', 'Avg_Age', 
                               'Subscription_Rate', 'Avg_Value_Score']
        
        print(tier_analysis)
        
        # 2. Platinum Customer Profile (Top Tier)
        print(f"\nPLATINUM CUSTOMER PROFILE:")
        platinum_customers = self.df[self.df['Value_Tier'] == 'Platinum']
        
        print(f"Platinum customers: {len(platinum_customers)} ({len(platinum_customers)/len(self.df)*100:.1f}%)")
        print(f"Revenue contribution: ${platinum_customers['Purchase Amount (USD)'].sum():,.2f} ({platinum_customers['Purchase Amount (USD)'].sum()/self.df['Purchase Amount (USD)'].sum()*100:.1f}%)")
        print(f"Average age: {platinum_customers['Age'].mean():.1f} years")
        print(f"Gender split: {platinum_customers['Gender'].value_counts(normalize=True).round(2).to_dict()}")
        print(f"Top categories: {platinum_customers['Category'].value_counts().head(3).to_dict()}")
        print(f"Top locations: {platinum_customers['Location'].value_counts().head(3).to_dict()}")
        print(f"Subscription rate: {(platinum_customers['Subscription Status'] == 'Yes').mean()*100:.1f}%")
        print(f"Average rating given: {platinum_customers['Review Rating'].mean():.2f}")
        print(f"Most common frequency: {platinum_customers['Frequency of Purchases'].mode().iloc[0]}")
        
        # 3. Value Tier Comparison
        print(f"\nVALUE TIER COMPARISON:")
        tier_comparison = pd.DataFrame({
            'Tier': ['Bronze', 'Silver', 'Gold', 'Platinum'],
            'Customer_Percentage': [
                (self.df['Value_Tier'] == tier).mean() * 100 
                for tier in ['Bronze', 'Silver', 'Gold', 'Platinum']
            ],
            'Revenue_Percentage': [
                self.df[self.df['Value_Tier'] == tier]['Purchase Amount (USD)'].sum() / 
                self.df['Purchase Amount (USD)'].sum() * 100 
                for tier in ['Bronze', 'Silver', 'Gold', 'Platinum']
            ]
        }).round(1)
        
        print(tier_comparison)
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('High-Value Customer Analysis', fontsize=16, fontweight='bold', y=0.98)
        
        # Define consistent colors for all charts
        tier_colors = {
            'Bronze': '#CD7F32',   # Bronze color
            'Silver': '#C0C0C0',   # Silver color  
            'Gold': '#FFD700',     # Gold color
            'Platinum': '#E5E4E2'  # Platinum color
        }
        
        # Create ordered color list
        ordered_colors = [tier_colors['Bronze'], tier_colors['Silver'], 
                         tier_colors['Gold'], tier_colors['Platinum']]
        
        # 1. Customer count by tier
        tier_counts = self.df['Value_Tier'].value_counts()
        # Reorder to match Bronze -> Platinum
        tier_order = ['Bronze', 'Silver', 'Gold', 'Platinum']
        tier_counts_ordered = tier_counts.reindex(tier_order)
        
        tier_counts_ordered.plot(kind='bar', ax=axes[0,0], color=ordered_colors)
        axes[0,0].set_title('Customer Count by Value Tier', pad=20)
        axes[0,0].set_ylabel('Number of Customers')
        axes[0,0].tick_params(axis='x', rotation=0)
        
        # 2. Revenue by tier
        tier_revenue = self.df.groupby('Value_Tier')['Purchase Amount (USD)'].sum()
        tier_revenue_ordered = tier_revenue.reindex(tier_order)
        
        tier_revenue_ordered.plot(kind='bar', ax=axes[0,1], color=ordered_colors)
        axes[0,1].set_title('Total Revenue by Value Tier', pad=20)
        axes[0,1].set_ylabel('Revenue (USD)')
        axes[0,1].tick_params(axis='x', rotation=0)
        
        # 3. Value score distribution
        self.df['Value_Score'].hist(bins=30, ax=axes[1,0], edgecolor='black', alpha=0.7, color='skyblue')
        axes[1,0].set_title('Customer Value Score Distribution', pad=20)
        axes[1,0].set_xlabel('Value Score')
        axes[1,0].set_ylabel('Frequency')
        
        # 4. Tier characteristics heatmap
        tier_chars = self.df.groupby('Value_Tier').agg({
            'Purchase Amount (USD)': 'mean',
            'Previous Purchases': 'mean',
            'Age': 'mean',
            'Review Rating': 'mean'
        })
        
        # Reorder to match our tier order
        tier_chars_ordered = tier_chars.reindex(tier_order)
        
        # Normalize for heatmap
        tier_chars_norm = (tier_chars_ordered - tier_chars_ordered.min()) / (tier_chars_ordered.max() - tier_chars_ordered.min())
        sns.heatmap(tier_chars_norm.T, annot=True, fmt='.2f', cmap='RdYlBu_r', ax=axes[1,1])
        axes[1,1].set_title('Tier Characteristics (Normalized)', pad=20)
        axes[1,1].set_xticklabels(axes[1,1].get_xticklabels(), rotation=0, ha='right')
        axes[1,1].set_yticklabels(axes[1,1].get_yticklabels(), rotation=45)
        
        # Adjust spacing to prevent overlap
        plt.subplots_adjust(hspace=0.35, wspace=0.3, top=0.93, bottom=0.1)
        plt.show()
        
    def purchase_behavior_pca_analysis(self):
        """
        Principal Component Analysis of Purchase Behavior
        Focus: Identify key factors influencing customer shopping behavior
        Goal: Understand what drives customer decisions and spending
        """
        print("\n" + "="*60)
        print("PURCHASE BEHAVIOR FACTOR ANALYSIS (PCA)")
        print("="*60)
        
        # Prepare data for PCA
        # Select relevant numerical and categorical variables
        pca_data = self.df.copy()
        
        # Create numerical features from categorical variables
        print("FEATURE ENGINEERING FOR PCA:")
        
        # Encode categorical variables
        pca_data['Gender_Male'] = (pca_data['Gender'] == 'Male').astype(int)
        pca_data['Subscription_Yes'] = (pca_data['Subscription Status'] == 'Yes').astype(int)
        pca_data['Discount_Applied'] = (pca_data['Discount Applied'] == 'Yes').astype(int)
        pca_data['Promo_Used'] = (pca_data['Promo Code Used'] == 'Yes').astype(int)
        
        # Encode frequency as numerical (higher = more frequent)
        frequency_mapping = {
            'Weekly': 7, 'Bi-Weekly': 6, 'Fortnightly': 5, 'Monthly': 4,
            'Quarterly': 3, 'Every 3 Months': 2, 'Annually': 1
        }
        pca_data['Frequency_Numeric'] = pca_data['Frequency of Purchases'].map(frequency_mapping)
        
        # Encode shipping type (Express = 1, others = 0 for speed preference)
        pca_data['Express_Shipping'] = (pca_data['Shipping Type'] == 'Express').astype(int)
        
        # Create category dummies for major categories
        top_categories = pca_data['Category'].value_counts().head(4).index
        for category in top_categories:
            pca_data[f'Category_{category}'] = (pca_data['Category'] == category).astype(int)
        
        # Select features for PCA
        pca_features = [
            'Age', 'Purchase Amount (USD)', 'Previous Purchases', 'Review Rating',
            'Gender_Male', 'Subscription_Yes', 'Discount_Applied', 'Promo_Used',
            'Frequency_Numeric', 'Express_Shipping'
        ] + [f'Category_{cat}' for cat in top_categories]
        
        # Clean data - remove any missing values
        pca_df = pca_data[pca_features].dropna()
        
        print(f"Features included in PCA: {len(pca_features)}")
        print(f"Samples for analysis: {len(pca_df)}")
        
        scaler = StandardScaler()
        pca_scaled = scaler.fit_transform(pca_df)
        
        # Perform PCA
        pca = PCA()
        pca_result = pca.fit_transform(pca_scaled)
        
        # Analyze results
        print(f"\nPCA RESULTS:")
        
        # Explained variance
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        print("Explained Variance by Component:")
        for i, (var, cum_var) in enumerate(zip(explained_variance[:6], cumulative_variance[:6])):
            print(f"   PC{i+1}: {var:.3f} ({var*100:.1f}%) - Cumulative: {cum_var:.3f} ({cum_var*100:.1f}%)")
        
        # Feature loadings for first 3 components
        print(f"\nFEATURE IMPORTANCE (Top Contributors to Each Component):")
        
        feature_names = pca_features
        
        for pc in range(min(3, len(explained_variance))):
            print(f"\nPrincipal Component {pc+1} (explains {explained_variance[pc]*100:.1f}% of variance):")
            
            # Get loadings (feature contributions)
            loadings = pca.components_[pc]
            
            # Create feature importance dataframe
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Loading': loadings,
                'Abs_Loading': np.abs(loadings)
            }).sort_values('Abs_Loading', ascending=False)
            
            # Show top 5 contributors
            top_features = feature_importance.head(5)
            for _, row in top_features.iterrows():
                direction = "positive" if row['Loading'] > 0 else "negative"
                print(f"   {row['Feature']}: {row['Loading']:.3f} ({direction} influence)")
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Purchase Behavior Factor Analysis (PCA)', fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Scree plot
        components = range(1, min(11, len(explained_variance)+1))
        axes[0,0].plot(components, explained_variance[:10], 'bo-')
        axes[0,0].set_title('Scree Plot - Explained Variance by Component', pad=20)
        axes[0,0].set_xlabel('Principal Component')
        axes[0,0].set_ylabel('Explained Variance Ratio')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Cumulative variance
        axes[0,1].plot(components, cumulative_variance[:10], 'ro-')
        axes[0,1].axhline(y=0.8, color='green', linestyle='--', label='80% Variance')
        axes[0,1].set_title('Cumulative Explained Variance', pad=20)
        axes[0,1].set_xlabel('Principal Component')
        axes[0,1].set_ylabel('Cumulative Variance Ratio')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. PC1 vs PC2 scatter plot
        axes[1,0].scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6, s=20)
        axes[1,0].set_title('Customer Distribution (PC1 vs PC2)', pad=20)
        axes[1,0].set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}% variance)')
        axes[1,0].set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}% variance)')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Feature loadings for PC1 and PC2
        pc1_loadings = pca.components_[0]
        pc2_loadings = pca.components_[1]
        distances = np.sqrt(pc1_loadings**2 + pc2_loadings**2)
        
        # Get top features by importance (distance from origin)
        important_features_idx = np.argsort(distances)[-10:]  # Top 10 most important
        
        # Among important features, select ones that are well-separated
        selected_features = []
        selected_positions = []
        min_distance_threshold = 0.1  # Minimum distance between selected points
        
        # Sort by importance (highest first)
        for idx in reversed(important_features_idx):
            x, y = pc1_loadings[idx], pc2_loadings[idx]
            
            # Check if this position is far enough from already selected positions
            too_close = False
            for sel_x, sel_y in selected_positions:
                distance = np.sqrt((x - sel_x)**2 + (y - sel_y)**2)
                if distance < min_distance_threshold:
                    too_close = True
                    break
            
            if not too_close:
                selected_features.append(idx)
                selected_positions.append((x, y))
                
            # Stop when we have enough well-separated features
            if len(selected_features) >= 6:
                break
        
        # If we don't have enough separated features, fill with most important ones
        if len(selected_features) < 6:
            remaining_needed = 6 - len(selected_features)
            for idx in reversed(important_features_idx):
                if idx not in selected_features:
                    selected_features.append(idx)
                    remaining_needed -= 1
                    if remaining_needed == 0:
                        break
        
        # Plot the selected features
        selected_x = pc1_loadings[selected_features]
        selected_y = pc2_loadings[selected_features]
        
        axes[1,1].scatter(selected_x, selected_y, s=80, c='red', alpha=0.7)
        
        # Add annotations for selected features
        for i, idx in enumerate(selected_features):
            x, y = pc1_loadings[idx], pc2_loadings[idx]
            feature_name = feature_names[idx]
            
            # Shorten long feature names
            if feature_name == 'Purchase Amount (USD)':
                feature_name = 'Purchase_Amount'
            elif feature_name == 'Previous Purchases':
                feature_name = 'Previous_Purchases'
            elif feature_name == 'Review Rating':
                feature_name = 'Review_Rating'
            elif feature_name == 'Frequency_Numeric':
                feature_name = 'Frequency'
            elif feature_name == 'Express_Shipping':
                feature_name = 'Express_Ship'
            elif feature_name.startswith('Category_'):
                feature_name = feature_name.replace('Category_', 'Type_')
            
            # Smart positioning based on quadrant and avoid overcrowding
            if x >= 0 and y >= 0:  # Top-right
                xytext = (15, 10)
                ha = 'left'
            elif x < 0 and y >= 0:  # Top-left
                xytext = (-15, 10)
                ha = 'right'
            elif x < 0 and y < 0:  # Bottom-left
                xytext = (-15, -15)
                ha = 'right'
            else:  # Bottom-right
                xytext = (15, -15)
                ha = 'left'
            
            axes[1,1].annotate(feature_name, (x, y),
                             xytext=xytext, textcoords='offset points', 
                             fontsize=9, ha=ha, va='center',
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
                             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1',
                                           color='black', lw=1))
        
        axes[1,1].set_title('Feature Loadings (PC1 vs PC2) - Key Features Only', pad=20)
        axes[1,1].set_xlabel('PC1 Loading')
        axes[1,1].set_ylabel('PC2 Loading')
        axes[1,1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[1,1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
        axes[1,1].grid(True, alpha=0.3)

        # Adjust spacing to prevent overlap
        plt.subplots_adjust(hspace=0.35, wspace=0.25, top=0.88, bottom=0.1)
        plt.show()
        
        # Business insights
        print(f"\nBUSINESS INSIGHTS FROM PCA:")
                # Find which components capture most variance
        main_components = np.where(explained_variance > 0.1)[0]  # Components explaining >10%
        
        if len(main_components) > 0:
            print(f"   Key behavioral factors: {len(main_components)} main components explain {cumulative_variance[len(main_components)-1]*100:.1f}% of variance")
            
        # Analyze top contributing features
        total_importance = np.sum(np.abs(pca.components_[:3]), axis=0)  # Sum of first 3 components
        top_overall_features = np.argsort(total_importance)[-5:]
        
        print(f"   Most influential factors overall:")
        for idx in reversed(top_overall_features):
            print(f"      - {feature_names[idx]}")
        return pca_result, feature_names, explained_variance
    
    def business_recommendations(self):
        """
        Generate actionable business recommendations based on analysis
        """
        print("\n" + "="*60)
        print("STRATEGIC BUSINESS RECOMMENDATIONS")
        print("="*60)
        
        # Calculate key metrics
        total_revenue = self.df['Purchase Amount (USD)'].sum()
        avg_order_value = self.df['Purchase Amount (USD)'].mean()
        total_customers = len(self.df)
        
        print("KEY PERFORMANCE INDICATORS:")
        print(f"   Total Revenue: ${total_revenue:,.2f}")
        print(f"   Average Order Value: ${avg_order_value:.2f}")
        print(f"   Total Customers: {total_customers:,}")
        print(f"   Average Customer Rating: {self.df['Review Rating'].mean():.2f}/5.0")
        
        # Category insights
        top_category = self.df.groupby('Category')['Purchase Amount (USD)'].sum().idxmax()
        best_season = self.df.groupby('Season')['Purchase Amount (USD)'].sum().idxmax()
        
        # Age group insights
        age_revenue = self.df.groupby(pd.cut(self.df['Age'], bins=[0, 25, 35, 50, 100], 
                                           labels=['18-25', '26-35', '36-50', '50+']))['Purchase Amount (USD)'].sum()
        top_age_group = age_revenue.idxmax()
        
        # Marketing insights
        discount_users = (self.df['Discount Applied'] == 'Yes').mean() * 100
        subscription_rate = (self.df['Subscription Status'] == 'Yes').mean() * 100
        
        # Frequency insights
        most_common_frequency = self.df['Frequency of Purchases'].value_counts().index[0]
        frequency_revenue = self.df.groupby('Frequency of Purchases')['Purchase Amount (USD)'].sum()
        top_revenue_frequency = frequency_revenue.idxmax()
        
        weekly_customers = (self.df['Frequency of Purchases'] == 'Weekly').sum()
        total_customers = len(self.df)
        weekly_percentage = (weekly_customers / total_customers) * 100
        
        print(f"\nSTRATEGIC RECOMMENDATIONS:")
        print(f"   1. PRODUCT STRATEGY:")
        print(f"      - Focus on {top_category} category (highest revenue)")
        print(f"      - Optimize inventory for {best_season} season")
        print(f"      - Expand offerings in high-performing categories")
        
        print(f"   2. CUSTOMER TARGETING:")
        print(f"      - Primary target: {top_age_group} age group (highest revenue)")
        print(f"      - Develop retention programs for high-value segments")
        print(f"      - Focus acquisition on similar demographic profiles")
        
        print(f"   3. MARKETING OPTIMIZATION:")
        print(f"      - Current discount usage: {discount_users:.1f}%")
        if discount_users < 50:
            print(f"      - Increase discount promotion to drive volume")
        else:
            print(f"      - Reduce discount dependency, focus on value")
        print(f"      - Subscription rate: {subscription_rate:.1f}% - room for growth")
        
        print(f"   4. PURCHASE FREQUENCY STRATEGY:")
        print(f"      - Most common frequency: {most_common_frequency}")
        print(f"      - Highest revenue frequency: {top_revenue_frequency}")
        print(f"      - Weekly shoppers: {weekly_percentage:.1f}% of customers")
        if weekly_percentage < 20:
            print(f"      - Opportunity to convert customers to more frequent shopping")
        print(f"      - Develop frequency-based loyalty programs")
        
        print(f"   5. OPERATIONAL IMPROVEMENTS:")
        avg_rating = self.df['Review Rating'].mean()
        if avg_rating < 4.0:
            print(f"      - Address product quality (avg rating: {avg_rating:.2f})")
        print(f"      - Optimize shipping options based on customer preferences")
        print(f"      - Implement loyalty program for repeat customers")
        
        print(f"\nNEXT STEPS:")
        print(f"   1. Implement customer segmentation strategy")
        print(f"   2. A/B test marketing campaigns by segment")
        print(f"   3. Create seasonal inventory planning")
        print(f"   4. Develop subscription growth initiatives")
        print(f"   5. Monitor KPIs with automated dashboard")
    
    def run_complete_analysis(self):
        """
        Execute the complete business analysis pipeline
        """
        if not self.load_and_explore_data():
            return
            
        self.data_quality_overview()
        self.customer_demographics_analysis()
        self.product_performance_analysis()
        self.customer_segmentation_analysis()
        self.purchase_behavior_pca_analysis()
        self.business_recommendations()
        

    def business_recommendations(self):
        """
        Generate actionable business recommendations based on analysis
        """
        print("\n" + "="*60)
        print("STRATEGIC BUSINESS RECOMMENDATIONS")
        print("="*60)
        
        # Calculate key metrics
        total_revenue = self.df['Purchase Amount (USD)'].sum()
        avg_order_value = self.df['Purchase Amount (USD)'].mean()
        total_customers = len(self.df)
        
        print("KEY PERFORMANCE INDICATORS:")
        print(f"   Total Revenue: ${total_revenue:,.2f}")
        print(f"   Average Order Value: ${avg_order_value:.2f}")
        print(f"   Total Customers: {total_customers:,}")
        print(f"   Average Customer Rating: {self.df['Review Rating'].mean():.2f}/5.0")
        
        # Category insights
        top_category = self.df.groupby('Category')['Purchase Amount (USD)'].sum().idxmax()
        best_season = self.df.groupby('Season')['Purchase Amount (USD)'].sum().idxmax()
        
        # Age group insights
        age_revenue = self.df.groupby(pd.cut(self.df['Age'], bins=[0, 25, 35, 50, 100], 
                                           labels=['18-25', '26-35', '36-50', '50+']))['Purchase Amount (USD)'].sum()
        top_age_group = age_revenue.idxmax()
        
        # Marketing insights
        discount_users = (self.df['Discount Applied'] == 'Yes').mean() * 100
        subscription_rate = (self.df['Subscription Status'] == 'Yes').mean() * 100
        
        # Frequency insights
        most_common_frequency = self.df['Frequency of Purchases'].value_counts().index[0]
        frequency_revenue = self.df.groupby('Frequency of Purchases')['Purchase Amount (USD)'].sum()
        top_revenue_frequency = frequency_revenue.idxmax()
        
        weekly_customers = (self.df['Frequency of Purchases'] == 'Weekly').sum()
        total_customers = len(self.df)
        weekly_percentage = (weekly_customers / total_customers) * 100
        
        print(f"\nSTRATEGIC RECOMMENDATIONS:")
        print(f"   1. PRODUCT STRATEGY:")
        print(f"      - Focus on {top_category} category (highest revenue)")
        print(f"      - Optimize inventory for {best_season} season")
        print(f"      - Expand offerings in high-performing categories")
        
        print(f"   2. CUSTOMER TARGETING:")
        print(f"      - Primary target: {top_age_group} age group (highest revenue)")
        print(f"      - Develop retention programs for high-value segments")
        print(f"      - Focus acquisition on similar demographic profiles")
        
        print(f"   3. MARKETING OPTIMIZATION:")
        print(f"      - Current discount usage: {discount_users:.1f}%")
        if discount_users < 50:
            print(f"      - Increase discount promotion to drive volume")
        else:
            print(f"      - Reduce discount dependency, focus on value")
        print(f"      - Subscription rate: {subscription_rate:.1f}% - room for growth")
        
        print(f"   4. PURCHASE FREQUENCY STRATEGY:")
        print(f"      - Most common frequency: {most_common_frequency}")
        print(f"      - Highest revenue frequency: {top_revenue_frequency}")
        print(f"      - Weekly shoppers: {weekly_percentage:.1f}% of customers")
        if weekly_percentage < 20:
            print(f"      - Opportunity to convert customers to more frequent shopping")
        print(f"      - Develop frequency-based loyalty programs")
        
        print(f"   5. OPERATIONAL IMPROVEMENTS:")
        avg_rating = self.df['Review Rating'].mean()
        if avg_rating < 4.0:
            print(f"      - Address product quality (avg rating: {avg_rating:.2f})")
        print(f"      - Optimize shipping options based on customer preferences")
        print(f"      - Implement loyalty program for repeat customers")
        
        print(f"\nNEXT STEPS:")
        print(f"   1. Implement customer segmentation strategy")
        print(f"   2. A/B test marketing campaigns by segment")
        print(f"   3. Create seasonal inventory planning")
        print(f"   4. Develop subscription growth initiatives")
        print(f"   5. Monitor KPIs with automated dashboard")

    def run_complete_analysis(self):
        """
        Execute the complete business analysis pipeline
        """
        if not self.load_and_explore_data():
            print('NO')
            return
        self.data_quality_overview()
        self.customer_demographics_analysis()
        self.product_performance_analysis()
        self.create_us_map_visualization()
        self.customer_segmentation_analysis()
        self.purchase_behavior_pca_analysis()
        self.business_recommendations()
if __name__ == "__main__":
    data_path = os.getcwd()
    analyzer = CustomerShoppingAnalyzer(data_path)
    analyzer.run_complete_analysis()
    