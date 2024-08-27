import pandas as pd

data = pd.read_excel('Mock.xlsx', sheet_name='Sheet1')

for row,entry in data.iterrows():
    print("Complaint:", entry['Complaint'])
    print("Summary:", entry['Summary'])
    print("AllegedParty:", entry['AllegedParty'])
    print("Accusation:", entry['Accusation'])
    print("Location:", entry['Location'])
    print("Amount:", entry['Amount'])
    print("Category_Assigned:", entry['Category_Assigned'])
    print("Category:", entry['Category'])
    print()
    print()