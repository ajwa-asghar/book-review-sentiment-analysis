import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_sentiment_distribution_chart(df, sentiment_column='sentiment'):
    """
    Create a bar chart showing the distribution of sentiments
    """
    plt.figure(figsize=(10, 6))
    sentiment_counts = df[sentiment_column].value_counts()
    
    # Create bar chart
    bars = plt.bar(sentiment_counts.index, sentiment_counts.values, 
                   color=['#2E8B57', '#FF6B6B', '#4ECDC4'])
    
    # Add value labels on top of bars
    for bar, count in zip(bars, sentiment_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(sentiment_counts.values),
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Distribution of Sentiments in Book Reviews', fontsize=16, fontweight='bold')
    plt.xlabel('Sentiment', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('sentiment_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Sentiment Distribution Chart saved as 'sentiment_distribution.png'")
    print(f"Total reviews: {len(df)}")
    print(f"Sentiment breakdown: {dict(sentiment_counts)}")

def create_confusion_matrix_visual(y_true, y_pred, class_names=None):
    """
    Create a confusion matrix visualization
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix - Sentiment Classification', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Calculate accuracy
    accuracy = (cm[0,0] + cm[1,1] + cm[2,2]) / cm.sum() if len(cm) == 3 else (cm[0,0] + cm[1,1]) / cm.sum()
    print(f"\nOverall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    print(f"Confusion Matrix saved as 'confusion_matrix.png'")

def create_combined_visualization(df, y_true, y_pred, sentiment_column='sentiment'):
    """
    Create both visualizations in one figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Sentiment distribution (left subplot)
    sentiment_counts = df[sentiment_column].value_counts()
    bars = ax1.bar(sentiment_counts.index, sentiment_counts.values, 
                   color=['#2E8B57', '#FF6B6B', '#4ECDC4'])
    
    for bar, count in zip(bars, sentiment_counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(sentiment_counts.values),
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Sentiment')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Confusion matrix (right subplot)
    cm = confusion_matrix(y_true, y_pred)
    class_names = sorted(df[sentiment_column].unique())
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax2, cbar_kws={'label': 'Count'})
    
    ax2.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('combined_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Combined visualization saved as 'combined_visualization.png'")

def create_word_cloud_visualization(df, sentiment_column='sentiment', text_column='processed_text'):
    """
    Create word cloud visualizations for different sentiments
    """
    try:
        from wordcloud import WordCloud
        
        # Get unique sentiments
        sentiments = df[sentiment_column].unique()
        
        # Create subplots
        fig, axes = plt.subplots(1, len(sentiments), figsize=(5*len(sentiments), 5))
        if len(sentiments) == 1:
            axes = [axes]
        
        colors = ['Greens', 'Reds', 'Blues']
        
        for i, sentiment in enumerate(sentiments):
            # Get text for this sentiment
            sentiment_text = ' '.join(df[df[sentiment_column] == sentiment][text_column].dropna())
            
            if sentiment_text.strip():  # Only create wordcloud if there's text
                # Create word cloud
                wordcloud = WordCloud(
                    width=800, 
                    height=400, 
                    background_color='white',
                    colormap=colors[i % len(colors)],
                    max_words=100
                ).generate(sentiment_text)
                
                axes[i].imshow(wordcloud, interpolation='bilinear')
                axes[i].set_title(f'Word Cloud - {sentiment}', fontweight='bold')
                axes[i].axis('off')
            else:
                axes[i].text(0.5, 0.5, f'No text data for {sentiment}', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'Word Cloud - {sentiment}', fontweight='bold')
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('word_clouds.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Word clouds saved as 'word_clouds.png'")
        
    except ImportError:
        print("WordCloud library not installed. Install with: pip install wordcloud")

# Example usage (uncomment and modify as needed):
"""
# Load your data
df = pd.read_csv('Book Reviews.csv')

# After training your model and getting predictions
# y_true = your_true_labels
# y_pred = your_predicted_labels

# Create visualizations
create_sentiment_distribution_chart(df)
create_confusion_matrix_visual(y_true, y_pred)
create_combined_visualization(df, y_true, y_pred)
create_word_cloud_visualization(df)
"""

if __name__ == "__main__":
    print("Visualization functions loaded successfully!")
    print("\nAvailable functions:")
    print("1. create_sentiment_distribution_chart(df, sentiment_column)")
    print("2. create_confusion_matrix_visual(y_true, y_pred, class_names)")
    print("3. create_combined_visualization(df, y_true, y_pred, sentiment_column)")
    print("4. create_word_cloud_visualization(df, sentiment_column, text_column)")
    print("\nImport this file in your notebook and call these functions!") 