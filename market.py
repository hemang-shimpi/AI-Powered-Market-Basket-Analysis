import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


# loading the dataset
def load_data(file_path):
    data = pd.read_csv(file_path)

    # cleaning the dataset
    df = data.dropna(subset=['Description'])
    df = df[~df['Invoice'].astype(str).str.startswith('C')]
    df = df[df['Quantity'] > 0]
    df["Description"] = df["Description"].str.strip().str.lower()

    top_products = df['Description'].value_counts().head(100).index
    df = df[df['Description'].isin(top_products)]

    # Check for missing Customer ID values
    missing_customer_ids = df['Customer ID'].isnull().sum()
    if missing_customer_ids > 0:
        print(f"Warning: There are {missing_customer_ids} missing CustomerID values.")

    return df

# creating the basket matrix
def basket_matrix(df):
    # creating the basket matrix
    basket = (df.groupby(['Invoice', 'Description'])['Quantity']).sum().unstack(fill_value=0)

    # converting the quantities to boolean values
    basket = basket.astype(bool)

    return basket

# applying the apriori algorithm
def apply_apriori(basket):
    frequent_itemsets = apriori(basket, min_support = 0.005, use_colnames=True)

    # adding itemset length column
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(len)

    # generating the rules
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

    # cleaning up the rules dataframe for better readability
    rules['antecedent_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequent_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

    # sorting the rules by lift
    rules = rules.sort_values(by='lift', ascending=False)

    # saving the results to csv files
    rules.to_csv('association_rules.csv', index=False)
    frequent_itemsets.to_csv('frequent_itemsets.csv', index=False)

    return rules

def main():
    # loading the market basket dataset here
    data = load_data('market_basket.csv')
    basket = basket_matrix(data) # basket matrix
    rules = apply_apriori(basket) # applying apriori algorithm
    print(rules[['antecedent_str', 'consequent_str', 'support', 'confidence', 'lift']].head(10))

if __name__ == "__main__":
    main()