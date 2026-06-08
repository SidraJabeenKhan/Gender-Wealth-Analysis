#!/usr/bin/env python3
# Gender Wealth Inequality Analysis - Sample Script
# Demonstrates data analysis skills for historical wealth research.
# Author: Sidra Jabeen Khan
# Date: June 2026

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sample dataset simulating historical estate inventory data
# Columns: id, year, gender, total_wealth, real_estate, personal_property, debts
data = {
    'id': range(1, 101),
    'year': np.random.choice(range(1850, 1951), 100),
    'gender': np.random.choice(['M', 'F'], 100, p=[0.6, 0.4]),
    'total_wealth': np.random.lognormal(8, 1.5, 100),
    'real_estate': np.random.lognormal(7, 1.2, 100),
    'personal_property': np.random.lognormal(6, 1.0, 100),
    'debts': np.random.lognormal(5, 0.8, 100)
}

df = pd.DataFrame(data)

# Data cleaning
df['net_wealth'] = df['total_wealth'] - df['debts']
df = df[df['net_wealth'] > 0]  # Remove negative net wealth

# Basic descriptive statistics by gender
print("=== DESCRIPTIVE STATISTICS BY GENDER ===")
gender_stats = df.groupby('gender')['net_wealth'].describe()
print(gender_stats)
print()

# Percentile analysis
print("=== PERCENTILE ANALYSIS ===")
for gender in ['M', 'F']:
    subset = df[df['gender'] == gender]['net_wealth']
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print(f"\n{gender} percentiles:")
    for p in percentiles:
        print(f"  {p}th: {np.percentile(subset, p):.2f}")

# Basic Gini coefficient calculation (simplified)
def gini_coefficient(x):
    # Calculate Gini coefficient for wealth inequality
    sorted_x = np.sort(x)
    n = len(x)
    cumsum = np.cumsum(sorted_x)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

male_wealth = df[df['gender'] == 'M']['net_wealth']
female_wealth = df[df['gender'] == 'F']['net_wealth']

print("\n=== INEQUALITY MEASURES ===")
print(f"Overall Gini: {gini_coefficient(df['net_wealth']):.4f}")
print(f"Male Gini: {gini_coefficient(male_wealth):.4f}")
print(f"Female Gini: {gini_coefficient(female_wealth):.4f}")

# Wealth ratio analysis
male_mean = male_wealth.mean()
female_mean = female_wealth.mean()
print(f"\nMale/Female mean wealth ratio: {male_mean/female_mean:.2f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Wealth distribution by gender
df.boxplot(column='net_wealth', by='gender', ax=axes[0,0])
axes[0,0].set_title('Net Wealth Distribution by Gender')
axes[0,0].set_xlabel('Gender')
axes[0,0].set_ylabel('Net Wealth')

# 2. Histogram comparison
axes[0,1].hist(male_wealth, bins=20, alpha=0.5, label='Male', color='blue')
axes[0,1].hist(female_wealth, bins=20, alpha=0.5, label='Female', color='red')
axes[0,1].set_title('Wealth Distribution Histogram')
axes[0,1].set_xlabel('Net Wealth')
axes[0,1].set_ylabel('Frequency')
axes[0,1].legend()

# 3. Mean wealth over time (binned by decade)
df['decade'] = (df['year'] // 10) * 10
decade_gender = df.groupby(['decade', 'gender'])['net_wealth'].mean().unstack()
decade_gender.plot(kind='bar', ax=axes[1,0])
axes[1,0].set_title('Mean Net Wealth by Decade and Gender')
axes[1,0].set_xlabel('Decade')
axes[1,0].set_ylabel('Mean Net Wealth')
axes[1,0].legend(title='Gender')

# 4. Wealth composition
wealth_comp = df.groupby('gender')[['real_estate', 'personal_property']].mean()
wealth_comp.plot(kind='bar', stacked=True, ax=axes[1,1])
axes[1,1].set_title('Wealth Composition by Gender')
axes[1,1].set_xlabel('Gender')
axes[1,1].set_ylabel('Mean Value')
axes[1,1].legend(title='Asset Type')

plt.tight_layout()
plt.savefig('gender_wealth_analysis.png', dpi=150)
plt.show()

print("\nAnalysis complete. Visualization saved as 'gender_wealth_analysis.png'")
