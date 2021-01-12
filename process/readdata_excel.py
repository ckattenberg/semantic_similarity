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
        excel_dfs[sheet].replace('', np.nan, inplace=True)
        # Remove empty rows
        excel_dfs[sheet].dropna(axis=0, how='all', inplace=True)
		# Remove empty columns
        excel_dfs[sheet].dropna(axis=1, how='all', inplace=True)

    # dict_keys(['Overzicht', 't q1', 'T1 q2', 'T1 q3', 't2 q1', 't2 q2', 't2 q3', 't3 q1', 't3 q2', 't3 q3', 'Corona'])
    return(excel_dfs)

# Returns a list of all relevant sentences in the dataset where every sentence is a string.
# Hope you like spaghetti.
def get_sentences(df_dict):
    exceptions = ["In hoeverre herken je deze quote (niet) als het gaat om je eigen ervaringen?", 
                    "Wat is er aan de hand?",
                    "Wat is er nodig?"]

    sentences = []
    # For each datafframe in the dictionary
    for sheet_name in df_dict.keys():
        # Skip the 'Overzicht' sheet
        if(sheet_name != 'Overzicht'):
            df = df_dict[sheet_name]

            # Loop over the columns (except for the first one)
            for column_name in df.columns[1:]:
                column = df[column_name]

                # For each row in the column, append the sentence to sencences, unless in exceptions.
                for index, value in column.iteritems():
                    # Skip NaN values
                    if(value != np.nan and not (pd.isnull(value))):
                        # Skip exeption sentences
                        if(value not in exceptions and isinstance(value, str)):
                            # Skip urls
                            if("https" not in value):
                                sentences.append(value)

    return(sentences)

def main():
    excel_dfs = read_excel()
    get_sentences(excel_dfs)

if __name__ == '__main__':
    main()
