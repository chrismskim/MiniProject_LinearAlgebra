import pandas as pd

#get datats from execl sheets
sheets = pd.read_excel(
    "data/raw/online_retail_II.xlsx",
    sheet_name=None
)


#Combine the two sheet(2009-2010 and 2010-2011)
df = pd.concat(sheets.values(), ignore_index=True)


#Make Pandas Table 
profile = pd.DataFrame({
    "Data Type": df.dtypes.astype(str),    #Data Type
    "Missing Values": df.isnull().sum(),   #Count Nan
    "Unique Values": df.nunique(dropna=True)  #Count how many different values in data
})

print("\nData Profile Table:")
print(profile)