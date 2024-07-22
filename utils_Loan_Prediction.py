# Import required Python libraries for the analysis
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings
import category_encoders as ce
from scipy import stats
from scipy.stats import pearsonr
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score


def plot_count_percentage_missing(data, column_name):
    """
    Function to plot a countplot with count, percentage of categories of a column and missing values in the data
    
    Args:
        data: dataset name
        column_name: name of the column to be plotted.

    Returns: 
        None(Plots of count, percentage and missing values in a column)
    
    """
    # Calculate value counts and percentages
    value_counts = data[column_name].value_counts()
    percentages = (value_counts / len(data[column_name])) * 100

    # Calculate missing values percentage
    missing_percentage = (data[column_name].isnull().sum() / len(data[column_name])) * 100

    # Create a countplot
    sns.set(style="darkgrid")
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=data, x=column_name)

    # Add count, percentage, and missing values percentage labels to the bars
    total = len(data[column_name])
    for p in ax.patches:
        count = int(p.get_height())
        percentage = '{:.1f}%'.format(100 * p.get_height() / total)
        label = f'{count} ({percentage})'
        x = p.get_x() + p.get_width() / 2 - 0.1
        y = p.get_height() + 0.5
        ax.annotate(label, (x, y), fontsize=10, ha='center')

    # Add missing values percentage to the title
    title = f'Count and Percentage of {column_name.capitalize()} (Missing Data: {missing_percentage:.1f}%)'
    plt.xlabel(column_name.capitalize(), fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

    plt.show()


def visualize_numeric_variables(data, numeric_columns):
    """
    Function to visualize numeric variables in the data with histogram, box plot and violin plot.
    
    Args:
        data: dataset name
        numeric_columns: names of the numeric columns to be plotted.

    Returns:
        None(Plots to visualize numeric variables in a dataset)
    
    """
    # Set the style for Seaborn plots
    sns.set(style="whitegrid")

    # Loop through each numeric column and create visualizations
    for column in numeric_columns:
        # Create a figure with subplots
        plt.figure(figsize=(12, 6))
        
        # Plot a histogram
        plt.subplot(2, 2, 1)
        sns.histplot(data=data, x=column, kde=True)
        plt.title(f'Histogram of {column}', fontsize=14)

        # Plot a box plot
        plt.subplot(2, 2, 2)
        sns.boxplot(data=data, y=column)
        plt.title(f'Box Plot of {column}', fontsize=14)

        # Plot a violin plot
        plt.subplot(2, 2, 3)
        sns.violinplot(data=data, y=column)
        plt.title(f'Violin Plot of {column}', fontsize=14)

        # Adjust subplot layout
        plt.tight_layout()

        # Show the plots
        plt.show()


def visualize_correlations(data):
    """
    Function to visualize correlations between different columns i.e. heatmap between numeric variables and chi square metric between categorical variables.
    
    Args:
        data: dataset name

    Returns:
        None(Visualize correlations betwwen different columns (separately for numeric and categorical columns))
    
    """
    # Separate numeric and categorical columns
    numeric_columns = data.select_dtypes(include=['number']).columns
    categorical_columns = data.select_dtypes(exclude=['number']).columns

    # Calculate and visualize correlations for numeric variables
    if len(numeric_columns) > 1:
        plt.figure(figsize=(10, 8))
        numeric_corr = data[numeric_columns].corr()
        sns.heatmap(numeric_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Heatmap (Numeric Variables)', fontsize=16)
        plt.show()

    # Calculate and visualize correlations for categorical variables
    if len(categorical_columns) > 1:
        plt.figure(figsize=(10, 8))
        categorical_corr = pd.DataFrame(index=categorical_columns, columns=categorical_columns)
        for col1 in categorical_columns:
            for col2 in categorical_columns:
                if col1 != col2:
                    crosstab = pd.crosstab(data[col1], data[col2])
                    chi2, _, _, _ = stats.chi2_contingency(crosstab)
                    categorical_corr.at[col1, col2] = chi2

        sns.heatmap(categorical_corr.astype(float), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Heatmap (Categorical Variables)', fontsize=16)
        plt.show()


def scatter_plot(data, x_column, y_column):
    """
    Function to plot a scatter plot for a pair of columns provided.
    
    Args:
        data: dataset name
        x_column: name of the column to be plotted on X axis.
        y_column: name of the column to be plotted on Y axis.

    Returns:
        None(Scatter plot for a pair of columns provided.)
    
    """
    # Create a scatter plot
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=data, x=x_column, y=y_column)
    plt.title(f'Scatter Plot: {x_column} vs. {y_column}', fontsize=16)
    plt.xlabel(x_column, fontsize=12)
    plt.ylabel(y_column, fontsize=12)
    plt.show()


def evaluate_model_performance(y_true, y_pred):
    """
    Function to evaluate a model performance by calculating Precision, Recall, F1 Score, Classification Report and Confusion Matrix
    
    Args:
        y_true: True values of test set
        y_pred: model predictions

    Returns:
        None(Performance Metrics)
    
    """
    # Calculate precision, recall, and F1-score
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Generate a classification report
    class_report = classification_report(y_true, y_pred)

    # Generate a confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create a heatmap for the confusion matrix
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', linewidths=0.5, xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
    plt.xlabel('Predicted', fontsize=9)
    plt.ylabel('Actual', fontsize=9)
    plt.title('Confusion Matrix', fontsize=9)

    # Display the performance metrics
    print(f'Precision: {precision:.3f}')
    print(f'Recall: {recall:.3f}')
    print(f'F1 Score: {f1:.3f}')
    print('\nClassification Report:\n', class_report)

    # Show the confusion matrix
    plt.show()