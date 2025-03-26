import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import numpy as np

# Set the style to a more academic look with minimal grid lines
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style("white")  # Remove background lines
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.grid'] = False  # Turn off grid by default

# Academic-friendly color palette
academic_colors = {
    'blue': '#4472C4',
    'red': '#C00000',
    'green': '#548235',
    'orange': '#ED7D31',
    'purple': '#7030A0',
    'teal': '#00B0F0',
    'gold': '#FFC000',
    'gray': '#7F7F7F',
    'light_blue': '#8EAADB',
    'light_red': '#FF9999',
    'light_green': '#A9D18E',
    'light_purple': '#B4A7D6',
    'light_gray': '#D9D9D9'
}


# Data loading and processing function
def load_and_process_data(csv_path):
    """Load the CSV file and process the data for analysis."""
    df = pd.read_csv(csv_path)

    # Clean up column values by stripping whitespace
    if 'Platform' in df.columns:
        df['Platform'] = df['Platform'].str.strip()
    if 'sentiment' in df.columns:
        df['sentiment'] = df['sentiment'].str.strip()
    if 'Time of Tweet' in df.columns:
        df['Time of Tweet'] = df['Time of Tweet'].str.strip()

    # Map 'Time of Tweet' to specific times
    time_mapping = {'morning': '08:00:00', 'noon': '12:00:00', 'night': '20:00:00'}
    df['Time'] = df['Time of Tweet'].map(time_mapping)

    # Create a Datetime column
    df['Datetime'] = pd.to_datetime(
        df['Year'].astype(str) + '-' +
        df['Month'].astype(str) + '-' +
        df['Day'].astype(str) + ' ' +
        df['Time']
    )

    # Create a Year-Month column for monthly trends
    df['Year-Month'] = df['Datetime'].dt.to_period('M')

    return df


# Modified plotting functions
def plot_sentiment_distribution(df, output_dir):
    """Plot and save the distribution of sentiments with numerical labels."""
    plt.figure(figsize=(8, 6), facecolor='white')
    # Count the sentiments
    sentiment_counts = df['sentiment'].value_counts().sort_index()
    # Set fixed order for sentiments
    sentiment_order = ['negative', 'neutral', 'positive']
    sentiment_counts = sentiment_counts.reindex(sentiment_order)

    # Academic appropriate colors
    colors = [academic_colors['red'], academic_colors['blue'], academic_colors['green']]

    # Create bar plot
    ax = sentiment_counts.plot(kind='bar', color=colors, edgecolor='black', linewidth=0.5)

    # Add numerical labels on top of each bar
    for i, v in enumerate(sentiment_counts):
        ax.text(i, v + 5, str(v), ha='center', fontsize=9, fontweight='bold')

    plt.title('Sentiment Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Sentiment', fontsize=12)
    plt.ylabel('Number of Posts', fontsize=12)
    plt.xticks(rotation=0)  # Keep labels horizontal
    # Add only y-axis grid for reference
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentiment_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_platform_distribution(df, output_dir):
    """Plot and save the distribution of platforms."""
    plt.figure(figsize=(8, 6), facecolor='white')
    # Count platforms
    platform_counts = df['Platform'].value_counts()

    # Create bar plot with platform-specific colors (academic style)
    platform_colors_dict = {
        'Twitter': academic_colors['blue'],
        'Facebook': academic_colors['light_blue'],
        'Instagram': academic_colors['purple']
    }
    colors = [platform_colors_dict.get(platform, academic_colors['gray']) for platform in platform_counts.index]

    ax = platform_counts.plot(kind='bar', color=colors, edgecolor='black', linewidth=0.5)

    # Add numerical labels on top of each bar
    for i, v in enumerate(platform_counts):
        ax.text(i, v + 5, str(v), ha='center', fontsize=9, fontweight='bold')

    plt.title('Platform Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Platform', fontsize=12)
    plt.ylabel('Number of Posts', fontsize=12)
    plt.xticks(rotation=0)  # Keep labels horizontal
    # Add only y-axis grid for reference
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'platform_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_posts_per_year(df, output_dir):
    """Plot and save the number of posts per year."""
    posts_per_year = df['Year'].value_counts().sort_index()
    plt.figure(figsize=(10, 6), facecolor='white')
    ax = posts_per_year.plot(
        kind='bar',
        color=academic_colors['teal'],
        edgecolor='black',
        linewidth=0.5
    )

    # Add numerical labels on top of each bar
    for i, v in enumerate(posts_per_year):
        ax.text(i, v + 5, str(v), ha='center', fontsize=9, fontweight='bold')

    plt.title('Number of Posts per Year', fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Posts', fontsize=12)
    plt.xticks(rotation=0)  # Keep labels horizontal
    # Add only y-axis grid for reference
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'posts_per_year.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_sentiment_trends(df, output_dir):
    """Plot and save sentiment trends over time."""
    plt.figure(figsize=(12, 6), facecolor='white')

    # Define academic appropriate colors for sentiments
    sentiment_colors = {
        'positive': academic_colors['green'],
        'negative': academic_colors['red'],
        'neutral': academic_colors['blue']
    }

    # Define line styles and markers for better distinction in academic papers
    line_styles = {
        'positive': 'solid',
        'negative': 'dashed',
        'neutral': 'dashdot'
    }

    markers = {
        'positive': 'o',
        'negative': 's',  # square
        'neutral': '^'  # triangle
    }

    for sentiment in ['positive', 'negative', 'neutral']:
        sentiment_data = df[df['sentiment'] == sentiment]['Year-Month'].value_counts().sort_index()
        sentiment_data.index = sentiment_data.index.astype(str)  # Convert Period to string
        sentiment_data.plot(
            kind='line',
            label=sentiment,
            linewidth=2,
            color=sentiment_colors[sentiment],
            linestyle=line_styles[sentiment],
            marker=markers[sentiment],
            markersize=5
        )

    plt.title('Sentiment Trends Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Number of Posts', fontsize=12)
    plt.legend(title='Sentiment', frameon=True, fancybox=False, edgecolor='black')
    plt.xticks(rotation=45)
    # Add only y-axis grid for reference
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentiment_trends.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_wordcloud_all(df, output_dir):
    """Generate and save a word cloud for all posts."""
    text = ' '.join(df['text'].astype(str))
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='Blues',  # Academic appropriate
        max_words=150,
        contour_width=1,
        contour_color='gray'
    ).generate(text)

    plt.figure(figsize=(10, 5), facecolor='white')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of All Posts', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'wordcloud_all.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_wordcloud_by_sentiment(df, output_dir):
    """Generate and save word clouds for each sentiment."""
    # Define color maps appropriate for academic papers
    color_maps = {
        'positive': 'Greens',
        'negative': 'Reds',
        'neutral': 'Blues'
    }

    # Define contour colors
    contour_colors = {
        'positive': academic_colors['light_green'],
        'negative': academic_colors['light_red'],
        'neutral': academic_colors['light_blue']
    }

    for sentiment in ['positive', 'negative', 'neutral']:
        text = ' '.join(df[df['sentiment'] == sentiment]['text'].astype(str))
        if not text.strip():
            print(f"No text data for {sentiment} sentiment. Skipping word cloud.")
            continue

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap=color_maps[sentiment],
            max_words=150,
            contour_width=1,
            contour_color=contour_colors[sentiment]
        ).generate(text)

        plt.figure(figsize=(10, 5), facecolor='white')
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for {sentiment.capitalize()} Sentiment', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'wordcloud_{sentiment}.png'), dpi=300, bbox_inches='tight')
        plt.close()


def plot_sentiment_by_platform(df, output_dir):
    """Plot and save sentiment distribution by platform."""
    plt.figure(figsize=(10, 6), facecolor='white')

    # Create crosstab of platform and sentiment
    platform_sentiment = pd.crosstab(df['Platform'], df['sentiment'])

    # Define academic appropriate colors for sentiments
    sentiment_colors = {
        'positive': academic_colors['green'],
        'negative': academic_colors['red'],
        'neutral': academic_colors['blue']
    }

    # Plot unstacked bars
    platform_sentiment.plot(
        kind='bar',
        stacked=False,
        color=[sentiment_colors[col] for col in platform_sentiment.columns],
        figsize=(10, 6),
        edgecolor='black',
        linewidth=0.5
    )

    # Add value labels on top of each bar
    for i, platform in enumerate(platform_sentiment.index):
        for j, sentiment in enumerate(platform_sentiment.columns):
            value = platform_sentiment.loc[platform, sentiment]
            plt.text(i + (j - 1) * 0.25, value + 2, str(value), ha='center', fontsize=9, fontweight='bold')

    plt.title('Sentiment Distribution by Platform', fontsize=14, fontweight='bold')
    plt.xlabel('Platform', fontsize=12)
    plt.ylabel('Number of Posts', fontsize=12)
    plt.legend(title='Sentiment', frameon=True, fancybox=False, edgecolor='black')
    plt.xticks(rotation=0)  # Keep labels horizontal
    # Add only y-axis grid for reference
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentiment_by_platform.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_sentiment_percentage_by_platform(df, output_dir):
    """Plot and save sentiment percentage by platform."""
    platform_sentiment_norm = pd.crosstab(df['Platform'], df['sentiment'], normalize='index') * 100

    # Define academic appropriate colors for sentiments
    sentiment_colors = {
        'positive': academic_colors['green'],
        'negative': academic_colors['red'],
        'neutral': academic_colors['blue']
    }

    plt.figure(figsize=(10, 6), facecolor='white')

    # Plot stacked bars with percentage
    ax = platform_sentiment_norm.plot(
        kind='bar',
        stacked=True,
        figsize=(10, 6),
        color=[sentiment_colors[col] for col in platform_sentiment_norm.columns],
        edgecolor='black',
        linewidth=0.5
    )

    # Add percentage labels on each segment
    for i, platform in enumerate(platform_sentiment_norm.index):
        # Calculate the cumulative height for positioning text
        cumulative_height = 0
        for sentiment in platform_sentiment_norm.columns:
            height = platform_sentiment_norm.loc[platform, sentiment]
            if height > 3:  # Only label if percentage is large enough
                # Position text in the middle of each segment
                y_position = cumulative_height + height / 2
                # Add annotation
                ax.text(
                    i, y_position,
                    f"{height:.1f}%",
                    ha='center', va='center',
                    fontsize=9, fontweight='bold',
                    color='black'  # Black text for better readability
                )
            cumulative_height += height

    plt.title('Sentiment Percentage by Platform', fontsize=14, fontweight='bold')
    plt.xlabel('Platform', fontsize=12)
    plt.ylabel('Percentage', fontsize=12)
    plt.legend(title='Sentiment', frameon=True, fancybox=False, edgecolor='black')
    plt.xticks(rotation=0)  # Keep labels horizontal
    # No grid for stacked bar chart
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentiment_percentage_by_platform.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_text_length_by_category(df, category, output_dir):
    """Plot and save the distribution of text lengths by a specified category."""
    df['text_length'] = df['text'].apply(len)

    plt.figure(figsize=(10, 6), facecolor='white')

    # Get unique category values
    unique_categories = df[category].unique()

    # Define academic appropriate color maps
    if category == 'sentiment':
        # Define colors for sentiments
        category_colors = {
            'positive': academic_colors['green'],
            'negative': academic_colors['red'],
            'neutral': academic_colors['blue']
        }
        # Create a palette with a default color for any unknown categories
        palette = {cat: category_colors.get(cat, academic_colors['gray']) for cat in unique_categories}
    else:  # For platform
        # Define colors for platforms
        platform_colors = {
            'Twitter': academic_colors['blue'],
            'Facebook': academic_colors['light_blue'],
            'Instagram': academic_colors['purple']
        }
        # Create a palette with a default color for any unknown platforms
        palette = {cat: platform_colors.get(cat, academic_colors['gray']) for cat in unique_categories}

    # Create boxplot
    sns.boxplot(x=category, y='text_length', data=df, palette=palette, width=0.6, linewidth=1)

    # Add mean markers and values
    for i, cat_value in enumerate(unique_categories):
        mean_length = df[df[category] == cat_value]['text_length'].mean()
        plt.text(i, mean_length + 5, f'Mean: {mean_length:.1f}', ha='center', fontsize=9, fontweight='bold')

    plt.title(f'Text Length Distribution by {category.capitalize()}', fontsize=14, fontweight='bold')
    plt.xlabel(category.capitalize(), fontsize=12)
    plt.ylabel('Text Length (characters)', fontsize=12)
    # Add only y-axis grid for reference
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'text_length_by_{category}.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_sentiment_platform_heatmap(df, output_dir):
    """Plot and save a heatmap of sentiment vs. platform."""
    sentiment_platform_crosstab = pd.crosstab(df['sentiment'], df['Platform'])

    plt.figure(figsize=(10, 6), facecolor='white')
    ax = sns.heatmap(
        sentiment_platform_crosstab,
        annot=True,
        cmap='Blues',  # Academic appropriate
        fmt='d',
        linewidths=0.5,
        cbar_kws={'label': 'Number of Posts'}
    )

    plt.title('Heatmap of Sentiment vs. Platform', fontsize=14, fontweight='bold')
    plt.xlabel('Platform', fontsize=12)
    plt.ylabel('Sentiment', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentiment_platform_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()


# New heatmap functions
def plot_year_month_heatmap(df, output_dir):
    """Plot and save a heatmap of posts by year and month."""
    year_month_crosstab = pd.crosstab(df['Year'], df['Month'])

    plt.figure(figsize=(12, 8), facecolor='white')
    ax = sns.heatmap(
        year_month_crosstab,
        annot=True,
        cmap='Blues',  # Academic appropriate
        fmt='d',
        linewidths=0.5,
        cbar_kws={'label': 'Number of Posts'}
    )

    plt.title('Heatmap of Posts by Year and Month', fontsize=14, fontweight='bold')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Year', fontsize=12)

    # Set month labels (1-based to match your data)
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Adjust ticks based on the actual columns present in the data
    month_positions = np.arange(len(year_month_crosstab.columns)) + 0.5
    month_labels_present = [month_labels[i - 1] if 1 <= i <= 12 else str(i) for i in year_month_crosstab.columns]
    plt.xticks(month_positions, month_labels_present, rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'year_month_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_platform_year_heatmap(df, output_dir):
    """Plot and save a heatmap of posts by platform and year."""
    platform_year_crosstab = pd.crosstab(df['Platform'], df['Year'])

    plt.figure(figsize=(10, 6), facecolor='white')
    ax = sns.heatmap(
        platform_year_crosstab,
        annot=True,
        cmap='Blues',  # Academic appropriate
        fmt='d',
        linewidths=0.5,
        cbar_kws={'label': 'Number of Posts'}
    )

    plt.title('Heatmap of Posts by Platform and Year', fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Platform', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'platform_year_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_sentiment_year_heatmap(df, output_dir):
    """Plot and save a heatmap of sentiment by year."""
    sentiment_year_crosstab = pd.crosstab(df['sentiment'], df['Year'])

    plt.figure(figsize=(10, 6), facecolor='white')
    ax = sns.heatmap(
        sentiment_year_crosstab,
        annot=True,
        cmap='Blues',  # Academic appropriate
        fmt='d',
        linewidths=0.5,
        cbar_kws={'label': 'Number of Posts'}
    )

    plt.title('Heatmap of Sentiment by Year', fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Sentiment', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentiment_year_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_day_time_heatmap(df, output_dir):
    """Plot and save a heatmap of posts by day of week and time of day."""
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    time_order = ['morning', 'noon', 'night']

    # Create crosstab with day of week and time of day
    day_time_crosstab = pd.crosstab(df['Datetime'].dt.day_name(), df['Time of Tweet'])

    # Reindex to ensure consistent order
    day_time_crosstab = day_time_crosstab.reindex(index=day_order, columns=time_order)

    plt.figure(figsize=(10, 8), facecolor='white')
    ax = sns.heatmap(
        day_time_crosstab,
        annot=True,
        cmap='Blues',  # Academic appropriate
        fmt='d',
        linewidths=0.5,
        cbar_kws={'label': 'Number of Posts'}
    )

    plt.title('Heatmap of Posts by Day of Week and Time of Day', fontsize=14, fontweight='bold')
    plt.xlabel('Time of Day', fontsize=12)
    plt.ylabel('Day of Week', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'day_time_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_day_sentiment_heatmap(df, output_dir):
    """Plot and save a heatmap of sentiment by day of week."""
    # Create crosstab with day of week and sentiment
    day_sentiment_crosstab = pd.crosstab(df['Datetime'].dt.day_name(), df['sentiment'])

    # Reindex for consistent order
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_sentiment_crosstab = day_sentiment_crosstab.reindex(day_order)

    plt.figure(figsize=(10, 8), facecolor='white')
    ax = sns.heatmap(
        day_sentiment_crosstab,
        annot=True,
        cmap='Blues',  # Academic appropriate
        fmt='d',
        linewidths=0.5,
        cbar_kws={'label': 'Number of Posts'}
    )

    plt.title('Heatmap of Sentiment by Day of Week', fontsize=14, fontweight='bold')
    plt.xlabel('Sentiment', fontsize=12)
    plt.ylabel('Day of Week', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'day_sentiment_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()


# Main function to run all plots
def main(csv_path, output_dir):
    """Main function to load data and generate all plots."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load and process data
    df = load_and_process_data(csv_path)

    # Print unique platform values to debug
    print("Unique platform values in dataset:", df['Platform'].unique())

    # Generate all plots
    plot_sentiment_distribution(df, output_dir)
    plot_platform_distribution(df, output_dir)
    plot_posts_per_year(df, output_dir)
    plot_sentiment_trends(df, output_dir)
    plot_wordcloud_all(df, output_dir)
    plot_wordcloud_by_sentiment(df, output_dir)
    plot_sentiment_by_platform(df, output_dir)
    plot_sentiment_percentage_by_platform(df, output_dir)
    plot_text_length_by_category(df, 'sentiment', output_dir)
    plot_text_length_by_category(df, 'Platform', output_dir)

    # Heatmaps
    plot_sentiment_platform_heatmap(df, output_dir)
    plot_year_month_heatmap(df, output_dir)
    plot_platform_year_heatmap(df, output_dir)
    plot_sentiment_year_heatmap(df, output_dir)
    plot_day_time_heatmap(df, output_dir)
    plot_day_sentiment_heatmap(df, output_dir)

    print(f"All plots have been saved to '{output_dir}'")


# Entry point
if __name__ == '__main__':
    # Update these paths to match your local setup
    csv_path = 'C:\\Users\\Daniel ZHAO\\AIBA\\Deep_Learning\\dlgp\\code___\\sentiment_analysis.csv'  # Replace with your CSV file path
    output_dir = 'C:\\Users\\Daniel ZHAO\\AIBA\\Deep_Learning\\dlgp\\code___\\plot'  # Replace with your output directory
    main(csv_path, output_dir)