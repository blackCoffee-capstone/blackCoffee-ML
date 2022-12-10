import pandas as pd
import json
import sys
from analyzer import Analyzer
from utils import Parser

input_file_path = sys.argv[1]
output_file_path = sys.argv[2]

my_parser = Parser()
df = my_parser.read_json_make_dataframe(input_path= input_file_path)

my_analyzer = Analyzer(df)
print(my_analyzer.list_of_weeks)


final_list = []

for i in my_analyzer.list_of_weeks :
    weekly_dict = {
        "week" : int(i),
        "ranks" : my_analyzer.calculate_trend_rank(i) 
    }
    
    final_list.append(weekly_dict)




with open(output_file_path, "w", encoding='utf8') as json_file:
        result_as_json = json.dumps(final_list, ensure_ascii=False)
        result_as_json = result_as_json.replace('NaN','null')
        result_as_json = result_as_json.replace('None','null')
        json_file.write(result_as_json)



