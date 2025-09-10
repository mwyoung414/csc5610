import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


class Mungler:
    def __init__(self, df):
        self.df = df.copy()

    def drop_columns(self, columns):
        self.df.drop(columns=columns, inplace=True)

    def fill_missing_with_mean(self, column):
        mean_value = self.df[column].mean()
        self.df[column].fillna(mean_value, inplace=True)

    def encode_categorical(self, column):
        self.df[column] = self.df[column].astype('category').cat.codes

    def get_dataframe(self):
        return self.df
    
    @staticmethod
    def categorical_visualizations(data:pd.DataFrame, cat_var:str, target_var:str) -> None:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=data, x=cat_var, y=target_var)
        plt.title(f'Box plot of {target_var} by {cat_var}')
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.stripplot(data=data, x=cat_var, y=target_var, jitter=True)
        plt.title(f'Strip plot of {target_var} by {cat_var}')
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.barplot(data=data, x=cat_var, y=target_var, estimator=np.mean, errorbar="sd")
        plt.title(f'Bar plot of {target_var} by {cat_var}')
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.pointplot(data=data, x=cat_var, y=target_var, estimator=np.mean, errorbar="sd")
        plt.title(f'Point plot of {target_var} by {cat_var}')
        plt.show()

        g = sns.displot(data=data, x=target_var, col=cat_var, kind='kde', fill=True)
        plt.show()

    @staticmethod
    def correlation_heatmap(data:pd.DataFrame) -> None:
        plt.figure(figsize=(12, 10))
        corr = data.corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
        plt.title('Correlation Heatmap')
        plt.show()