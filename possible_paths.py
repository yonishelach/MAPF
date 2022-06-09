import pandas as pd
import glob
import random

def main():
    pass
    df=pd.read_csv("database/routing_requests_without_time.csv")
    rows = []
    for j in range(10):
        for i in range(30):
            src = df.iloc[j][f'source_{i}']
            dst = df.iloc[j][f'destination_{i}']
            # print(src, dst)
            filenames = glob.glob(f'database/paths_by_routing_request/src_{src}_dst_{dst}/*')
            filename = random.choice(filenames)
            paths_df = pd.read_csv(filename)
            chosen_row_idx = random.randint(0, len(paths_df)-1)
            row = paths_df.iloc[chosen_row_idx]
            rows.append(row.to_frame().T)
    new_df = pd.concat(rows, ignore_index=True).reset_index().drop(columns=['index'])
    new_df.to_csv('1.csv', index=False)





if __name__ == '__main__':
    main()
