import json, sys
import pandas as pd

def main():

    df = pd.read_json('./query.json', orient = 'records')
    df.to_json('moviequery.json.gz', orient='records', lines=True, compression='gzip')
    print('done')
    
    return

if __name__ == '__main__':
    main()