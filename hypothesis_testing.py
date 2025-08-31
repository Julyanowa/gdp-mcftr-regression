from math import sqrt
from descriptive_statistics import StatsAnalyzer
from scipy.stats import f

class HypothesisTests:
    @staticmethod
    def t_test(t_emp, df):
        t_table = {
            1:12.71, 2:4.303, 3:3.182, 4:2.776, 5:2.571,
            6:2.447, 7:2.365, 8:2.306, 9:2.262, 10:2.228,
            12:2.179, 15:2.131, 20:2.086, 25:2.060, 30:2.042,
            40:2.021, 60:2.000, 120:1.980
        }
        keys = sorted(t_table.keys())
        closest_df = max([k for k in keys if k <= df])
        t_cr = t_table[closest_df]
        return t_cr, t_emp > t_cr

    @staticmethod
    def f_test(F_emp, df_model, df_resid, alpha=0.05):
        F_cr = f.ppf(1 - alpha, df_model, df_resid)
        return F_cr, F_emp > F_cr

    @staticmethod
    def t_test_correlation(data_x, data_y):
        if len(data_x) != len(data_y):
            raise ValueError("Выборки должны быть одинаковой длины")
        n = len(data_x)
        r = StatsAnalyzer(data_x).pearson_corr(data_y)
        t_emp = abs(r) * sqrt(n - 2) / sqrt(1 - r**2)
        df = n - 2

        # таблица для t-критического
        t_table = {
            1:12.71, 2:4.303, 3:3.182, 4:2.776, 5:2.571,
            6:2.447, 7:2.365, 8:2.306, 9:2.262, 10:2.228,
            12:2.179, 15:2.131, 20:2.086, 25:2.060, 30:2.042,
            40:2.021, 60:2.000, 120:1.980
        }
        keys = sorted(t_table.keys())
        closest_df = max([k for k in keys if k <= df])
        t_cr = t_table[closest_df]
        return t_emp, t_cr, t_emp > t_cr