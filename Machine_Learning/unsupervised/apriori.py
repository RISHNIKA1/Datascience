import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder

dataset =[
    ['milk', 'bread', 'butter'],
    ['bread', 'butter'],
    ['milk', 'bread'],
    ['milk', 'butter'],
    ['bread']
]

#one hot encoding

te = TransactionEncoder()
te_ary = te.fit_transform(dataset)

df =pd.DataFrame(te_ary,columns =te.columns_)

#apriori
freq_times = apriori(df,min_support=0.4,use_colnames=True)
print(freq_times)

#association rules
rules = association_rules(freq_times,metric ="confidence",min_threshold=0.6)
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])