import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_default_distribution(df):
    """Plot distribution of default payments"""
    df1 = df.copy()
    df1['default payment next month'] = df1['default payment next month'].map({0: 'Normal', 1: 'Default'})
    
    plt.figure(figsize=(8, 7))
    ax = sns.countplot(x='default payment next month', data=df1, hue='default payment next month')
    plt.title('Distribution of Default Payment Next Month')
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    
    plt.show()

def plot_age_distribution(df):
    """Plot age distribution"""
    plt.figure(figsize=(12, 6))
    sns.histplot(df['AGE'], bins=20, kde=True)
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.show()

def plot_age_by_default(df):
    """Plot age distribution by default status"""
    df1 = df.copy()
    df1['default payment next month'] = df1['default payment next month'].map({0: 'Normal', 1: 'Default'})
    
    plt.figure(figsize=(10, 10))
    ax = sns.boxplot(x="default payment next month", y="AGE", data=df1, hue='default payment next month')
    plt.title('Age Distribution by Default Payment Next Month')
    plt.xlabel('Default Payment Next Month')
    plt.ylabel('Age')
    plt.show()

def plot_education_distribution(df):
    """Plot education distribution"""
    df1 = df.copy()
    df1['EDUCATION'] = df1['EDUCATION'].map({1: 'Graduate School', 2: 'University', 3: 'High School', 4: 'Others'})
    
    fig, axes = plt.subplots(ncols=2, figsize=(13, 8))
    df1['EDUCATION'].value_counts().plot(kind='pie', ax=axes[0], subplots=True)
    sns.countplot(x='EDUCATION', hue='default payment next month', data=df1, ax=axes[1])
    plt.tight_layout()
    plt.show()

def plot_gender_distribution(df):
    """Plot gender distribution"""
    df1 = df.copy()
    df1['SEX'] = df1['SEX'].map({1: 'Male', 2: 'Female'})
    
    fig, axes = plt.subplots(ncols=2, figsize=(13, 8))
    df1['SEX'].value_counts().plot(kind='pie', ax=axes[0], subplots=True)
    sns.countplot(x='SEX', hue='default payment next month', data=df1, ax=axes[1])
    plt.tight_layout()
    plt.show()

def plot_marriage_distribution(df):
    """Plot marriage status distribution"""
    df1 = df.copy()
    df1['MARRIAGE'] = df1['MARRIAGE'].map({1: 'Married', 2: 'Single', 3: 'Others'})
    
    fig, axes = plt.subplots(ncols=2, figsize=(13, 8))
    df1['MARRIAGE'].value_counts().plot(kind='pie', ax=axes[0], subplots=True)
    sns.countplot(x='MARRIAGE', hue='default payment next month', data=df1, ax=axes[1])
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df, k=10):
    """Plot correlation heatmap for top k features correlated with target"""
    # Convert all columns to numeric
    numeric_df = df.apply(pd.to_numeric, errors='coerce')
    numeric_df = numeric_df.dropna()
    
    # Calculate correlation
    corrmat = numeric_df.corr()
    cols = corrmat.nlargest(k, 'default payment next month')['default payment next month'].index
    cm = np.corrcoef(numeric_df[cols].values.T)
    
    # Plot heatmap
    sns.set(font_scale=1.25)
    plt.subplots(figsize=(10, 10))
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', 
                     annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    plt.show()
