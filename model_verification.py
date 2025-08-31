from descriptive_statistics import StatsAnalyzer
from math import sqrt


class RegressionVerifier:
    # --- таблица t-Стьюдента (alpha=0.05, двусторонний критерий) ---
    t_table = {
        1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        12: 2.179, 15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042,
        40: 2.021, 60: 2.000, 120: 1.980,
    }

    # --- таблица F-Фишера (alpha=0.05, df1=1, df2 разные) ---
    f_table = {
        1: 161.4, 2: 18.5, 3: 10.1, 4: 7.71, 5: 6.61,
        6: 5.99, 7: 5.59, 8: 5.32, 9: 5.12, 10: 4.96,
        12: 4.75, 15: 4.54, 20: 4.35, 25: 4.24, 30: 4.17,
        40: 4.08, 60: 4.00, 120: 3.92,
    }

    def __init__(self, x, y):
        if len(x) != len(y):
            raise ValueError("Размеры выборок X и Y должны совпадать")

        self.x = x
        self.y = y
        self.n = len(x)

        # --- коэффициенты регрессии ---
        stats_x = StatsAnalyzer(x)
        stats_y = StatsAnalyzer(y)
        cov_xy = stats_x.covariance(y)
        var_x = stats_x.variance()

        self.b = cov_xy / var_x                      # наклон (slope)
        self.a = stats_y.mean() - self.b * stats_x.mean()   # свободный член (intercept)

        # прогнозные значения
        self.y_hat = [self.a + self.b * xi for xi in x]

        # остатки
        self.residuals = [yi - yhi for yi, yhi in zip(y, self.y_hat)]

    # --- суммы квадратов ---
    def tss(self):
        mean_y = sum(self.y) / self.n
        return sum((yi - mean_y) ** 2 for yi in self.y)

    def rss(self):
        return sum(ei ** 2 for ei in self.residuals)

    def ess(self):
        return self.tss() - self.rss()

    # --- коэффициенты качества ---
    def r_squared(self):
        return self.ess() / self.tss()

    def variance_estimate(self):
        return self.rss() / (self.n - 2)

    # --- стандартные ошибки ---
    def standard_errors(self):
        mean_x = sum(self.x) / self.n
        s2 = self.variance_estimate()
        s_xx = sum((xi - mean_x) ** 2 for xi in self.x)

        se_b = sqrt(s2 / s_xx)
        se_a = sqrt(s2 * (1/self.n + mean_x**2/s_xx))

        return se_a, se_b

    # --- критерий Стьюдента ---
    def t_values(self):
        se_a, se_b = self.standard_errors()
        t_a = self.a / se_a
        t_b = self.b / se_b
        return t_a, t_b

    def t_critical(self):
        df = self.n - 2
        keys = sorted(self.t_table.keys())
        closest_df = max([k for k in keys if k <= df])
        return self.t_table[closest_df]

    def check_parameters_significance(self):
        t_a, t_b = self.t_values()
        t_cr = self.t_critical()
        return {
            "a": (t_a, abs(t_a) > t_cr),
            "b": (t_b, abs(t_b) > t_cr)
        }

    # --- критерий Фишера ---
    def f_test(self):
        return (self.ess() / 1) / (self.rss() / (self.n - 2))

    def f_critical(self):
        df2 = self.n - 2
        keys = sorted(self.f_table.keys())
        closest_df = max([k for k in keys if k <= df2])
        return self.f_table[closest_df]

    def check_model_significance(self):
        f_emp = self.f_test()
        f_cr = self.f_critical()
        return f_emp, f_emp > f_cr


# --- тест ---
if __name__ == "__main__":
    x = [1, 2, 4, 3, 5, 3, 4]
    y = [3, 7, 15, 10, 17, 10, 15]

    model = RegressionVerifier(x, y)

    print("✅ Коэффициенты регрессии:")
    print(f"a (intercept): {model.a:.3f}")
    print(f"b (slope): {model.b:.3f}")

    print("\n📊 Качество модели:")
    print(f"TSS = {model.tss():.3f}")
    print(f"ESS = {model.ess():.3f}")
    print(f"RSS = {model.rss():.3f}")
    print(f"R^2 = {model.r_squared():.3f}")

    print("\n📉 Стандартные ошибки:")
    se_a, se_b = model.standard_errors()
    print(f"SE(a) = {se_a:.3f}, SE(b) = {se_b:.3f}")

    print("\n🧪 Проверка значимости параметров:")
    t_a, t_b = model.t_values()
    print(f"t(a) = {t_a:.3f}, t(b) = {t_b:.3f}")
    print("Критическое t =", model.t_critical())
    signif = model.check_parameters_significance()
    print("a значим?" , "✅" if signif["a"][1] else "❌")
    print("b значим?" , "✅" if signif["b"][1] else "❌")

    print("\n🧪 Проверка значимости всей модели:")
    f_emp, f_ok = model.check_model_significance()
    print(f"F-эмп = {f_emp:.3f}, F-крит = {model.f_critical()}")
    print("Модель значима?" , "✅" if f_ok else "❌")