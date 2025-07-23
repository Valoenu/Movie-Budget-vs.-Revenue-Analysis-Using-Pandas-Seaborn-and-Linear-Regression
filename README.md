# 🎬 Movie Budget vs. Revenue Analysis

This project was created by a **software development student** exploring **Data Analysis with Python**. It investigates the relationship between movie production budgets and worldwide revenue using real-world data. The goal is to apply data wrangling, visualization, and regression modeling techniques in a practical project.
This project was build with AppBrewery Course

---

## 📂 Dataset

**Filename:** `cost_revenue_dirty.csv`

The dataset contains financial and release information for a large number of films, including:

- `USD_Production_Budget`
- `USD_Worldwide_Gross`
- `USD_Domestic_Gross`
- `Release_Date`

---

## 📊 Project Features

- **Data Cleaning**:
  - Removed duplicates and missing values
  - Converted financial strings (with `$` and `,`) to numeric types
  - Parsed and converted release dates
- **Exploratory Data Analysis**:
  - Summary statistics
  - Identified zero-revenue and unreleased films
- **Visualization**:
  - Scatterplots using Seaborn and Matplotlib
  - Visual encoding with color, size, and axes
- **Regression Modeling**:
  - Built univariate linear regression models using Scikit-learn
  - Compared old vs. new movies
  - Used model to estimate potential movie revenue

---

## 🧰 Libraries Used

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

---

## 📈 Example Outputs

- Top 5 most expensive new movies
- Percentage of movies that lost money
- Regression model R² scores
- Estimated revenue from a custom budget input

---

## 💡 Key Insight

> “There is a general positive correlation between movie budgets and global revenue — but a bigger budget doesn’t always guarantee success.”

---

## 🚀 Getting Started

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/movie-budget-revenue-analysis.git
