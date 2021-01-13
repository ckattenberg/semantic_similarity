import pandas as pd
from tabulate import tabulate
import numpy as np

# Returns a dictionary of dataframes. One for each sheet.
def read_excel(path="data/gebiedsmakelaars.xlsx"):
    # Use the pandas library to read in the excel file
    # Setting the sheet_name parameter to None makes it read in
    # all sheets in the file

    # The openpyxl engine supports .xlsx files
    excel_dfs = pd.read_excel(path, sheet_name=None, engine='openpyxl', header=None)

    # Remove all rows and columns that are completely empty.
    for sheet in excel_dfs.keys():
        excel_dfs[sheet].replace('', np.nan, inplace=True)
        # Remove empty rows
        excel_dfs[sheet].dropna(axis=0, how='all', inplace=True)
		# Remove empty columns
        excel_dfs[sheet].dropna(axis=1, how='all', inplace=True)

    # dict_keys(['Overzicht', 't q1', 'T1 q2', 'T1 q3', 't2 q1', 't2 q2', 't2 q3', 't3 q1', 't3 q2', 't3 q3', 'Corona'])
    return(excel_dfs)

def splitter(key, excel):
    '''Function takes a key, which is a document name, and the whole excel data sheet.
    
    Returns uniques and dups. Both are multi-dimensional lists. Uniques is a 2-dimensional
    list, each list inside of it contains the statements related to a certain question.
    Dups is 3 dimensional, it contains a list for each question, which contains lists containing
    statements that are the same. '''
    uniques = []
    dups = []
    sheet = excel[key]
    aspect_uniques = []
    aspect_dups = []
    for i in range(2, len(sheet)):
        try:
            if sheet[0][i] == "Stapeltje?":
                uniques.append(aspect_uniques)
                dups.append(aspect_dups)
                aspect_uniques = []
                aspect_dups = []
                continue
        
            if pd.isna(sheet[0][i]):
                aspect_uniques.append(sheet[1][i])
            else:
                entry = []
                for y in range(sheet[0][i] + 1):
                    entry.append(sheet[y+1][i])
                aspect_dups.append(entry)
        except:
            continue
    return uniques, dups
    
def main():
    excel_dfs = read_excel()
    keys = []
    for x in excel_dfs.keys():
        keys.append(x)
    uniques, dups = splitter(keys[1], excel_dfs)
    for x in uniques:
        print(x)

if __name__ == '__main__':
    main()
