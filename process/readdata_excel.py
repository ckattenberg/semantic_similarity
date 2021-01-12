import pandas as pd
from tabulate import tabulate
import numpy as np

# Returns a dictionary of dataframes. One for each sheet.
def read_excel(path="data/gebiedsmakelaars.xlsx"):
	# Use the pandas library to read in the excel file
	# Setting the sheet_name parameter to None makes it read in
	# all sheets in the file

	# The openpyxl engine supports .xlsx files
	excel_dfs = pd.read_excel(path, sheet_name=None, engine='openpyxl')

	# Remove all rows and columns that are completely empty.
	for sheet in excel_dfs.keys():
		print(sheet)
		excel_dfs[sheet].replace('', np.nan, inplace=True)
		excel_dfs[sheet].dropna(axis=0, how='all', inplace=True)

	# dict_keys(['Overzicht', 't q1', 'T1 q2', 'T1 q3', 't2 q1', 't2 q2', 't2 q3', 't3 q1', 't3 q2', 't3 q3', 'Corona'])
	return(excel_dfs)

def main():
	excel_dfs = read_excel()
	print(type(excel_dfs))
	print(type(excel_dfs['T1 q2']))
	df = excel_dfs['T1 q2']

	print(df.shape)

	print(df)


if __name__ == '__main__':
	main()
