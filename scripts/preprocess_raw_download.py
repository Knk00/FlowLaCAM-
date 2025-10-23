import pandas as pd
import glob, os

if __name__ == "__main__":
    DATA_DIR = "./data/raw/"
    RESULTS_DIR = DATA_DIR + 'results/'
    COMBINED_RESULTS_DIR = DATA_DIR + 'combined_results/'
    os.makedirs(COMBINED_RESULTS_DIR, exist_ok=True)
    MAP_DIR = DATA_DIR + 'maps/'
    MAPS = glob.glob(MAP_DIR + "*.map")
    MAP_NAMES = [m.replace('\\', '/').split('/')[-1].split('.map')[0] for m in MAPS]

    for map_name in MAP_NAMES:
        print(f"Processing {map_name}...")
        map_results = glob.glob(RESULTS_DIR + f'{map_name}*.csv')

        result_df = pd.DataFrame()

        for res_file in map_results:
            df = pd.read_csv(res_file)
            result_df = pd.concat([result_df, df], ignore_index=True)

        # Save the combined results
        result_df.to_csv(COMBINED_RESULTS_DIR + f'{map_name}.csv', index=False)