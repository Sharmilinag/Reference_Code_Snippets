import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from statistics import mean

new_file = 'crx.data'
old_file = 'credit.data'


def main():
    new_df = pd.read_csv(new_file)
    old_df = pd.read_csv(old_file)
    new_cols = new_df.columns.values
    old_cols = old_df.columns.values
    res = []
    for n_col in list(new_cols):
        for o_col in list(old_cols):
            row = {
                'old_tab': o_col,
                'new_tab': n_col,
                'col_name_match_score': fuzz.ratio(o_col, n_col)
            }
            unq_n = new_df[n_col].unique()
            unq_o = old_df[o_col].unique()
            row['new_tab_unique_cnt'] = len(unq_n)
            row['old_tab_unique_cnt'] = len(unq_o)
            best_match_percentage = 0.0
            if row['new_tab_unique_cnt'] < row['old_tab_unique_cnt']:
                best_match_percentage = row['new_tab_unique_cnt']*1.0/(
                        row['new_tab_unique_cnt']*row['old_tab_unique_cnt']) * 100.0
            else:
                best_match_percentage = row['old_tab_unique_cnt'] * 1.0 / (
                            row['new_tab_unique_cnt'] * row['old_tab_unique_cnt']) * 100.0
            row['best_match_percentage'] = round(best_match_percentage, 2)
            over_all_score = []
            score_85_abv = []
            score_50_85 = []
            score_20_50 = []
            score_0_20 = []
            for val_n in unq_n:
                for val_o in unq_o:
                    if not isinstance(val_o, str):
                        val_o = str(float(val_o))
                    if not isinstance(val_n, str):
                        val_n = str(float(val_n))
                    if val_n == val_o:
                        over_all_score.append(100)
                        score_85_abv.append(100)
                    else:
                        fuzzy_score = fuzz.ratio(val_n, val_o)
                        over_all_score.append(fuzzy_score)
                        if fuzzy_score > 85:
                            score_85_abv.append(fuzzy_score)
                        elif fuzzy_score > 50:
                            score_50_85.append(fuzzy_score)
                        elif fuzzy_score > 20:
                            score_20_50.append(fuzzy_score)
                        else:
                            score_0_20.append(fuzzy_score)
            if len(score_85_abv) == 0:
                mean_85_abv = 0
            else:
                mean_85_abv = mean(score_85_abv)
            if len(score_50_85) == 0:
                mean_50_85 = 0
            else:
                mean_50_85 = mean(score_50_85)
            if len(score_20_50) == 0:
                mean_20_50 = 0
            else:
                mean_20_50 = mean(score_20_50)
            if len(score_0_20) == 0:
                mean_0_20 = 0
            else:
                mean_0_20 = mean(score_0_20)
            row['over_all_avg'] = mean(over_all_score)
            row['85_100_avg'] = mean_85_abv
            row['85_100_percentage'] = round(((len(score_85_abv) * 1.0 / len(over_all_score)) * 100), 2)
            row['50_85_avg'] = mean_50_85
            row['50_85_percentage'] = round(((len(score_50_85) * 1.0 / len(over_all_score)) * 100), 2)
            row['20_50_avg'] = mean_20_50
            row['20_50_percentage'] = round(((len(score_20_50) * 1.0 / len(over_all_score)) * 100), 2)
            row['0_20_avg'] = mean_0_20
            row['0_20_percentage'] = round(((len(score_0_20) * 1.0 / len(over_all_score)) * 100), 2)
            res.append(row)

    res_df = pd.DataFrame(res)
    res_df.to_csv('match_analysis.csv', index=False)


if __name__ == "__main__":
    main()
