import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


# loading the dataset
def load_data(file_path):
    data = pd.read_csv(file_path)

    # cleaning the dataset
    df = data.dropna(subset=['Description'])
    df = df[~df['Invoice'].astype(str).str.startswith('C')]
    df[df['Quantity'] > 0]
    df["Description"] = df["Description"].str.strip().str.lower()

    # Check for missing Customer ID values
    missing_customer_ids = data['Customer ID'].isnull().sum()
    if missing_customer_ids > 0:
        print(f"Warning: There are {missing_customer_ids} missing CustomerID values.")

    return df

# creating the basket matrix
def basket_matrix(df):
    # creating the basket matrix
    basket = (df.groupby(['Invoice', 'Description'])['Quantity']).sum().unstack(fill_value=0)

    # converting the quantities to 1/0
    basket = basket.map(lambda x: 1 if x > 0 else 0)

    return basket


def main():
    # loading the market basket dataset here
    data = load_data('market_basket.csv')
    basket = basket_matrix(data) # basket matrix


if __name__ == "__main__":
    main()