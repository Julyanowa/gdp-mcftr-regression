from descriptive_statistics import StatsAnalyzer
from hypothesis_testing import HypothesisTests
from model_verification import RegressionVerifier
import pandas as pd
import matplotlib.pyplot as plt

# === 1. Импорт данных ===
data = pd.read_csv(r"D:\pgAdmin1\gdp_value_mcftr.csv", sep=";")
data.columns = data.columns.str.strip()

# --- Приведение дат к datetime ---
if "date" in data.columns:
    data["date"] = pd.to_datetime(data["date"], dayfirst=True)

# --- Определяем X и Y ---
X = data["gdp_value"].astype(float).tolist()
Y = data["mcftr_index"].astype(float).tolist()

print("✅ Данные успешно загружены\n")


# === 2. Анализ модели ===
def analyze_model(X, Y, label):
    print(f"{'='*50}\n📊 {label}\n{'='*50}")
    
    stats_x = StatsAnalyzer(X)
    stats_y = StatsAnalyzer(Y)
    print(f"X mean={stats_x.mean():.3f}, std={stats_x.std_dev():.3f}")
    print(f"Y mean={stats_y.mean():.3f}, std={stats_y.std_dev():.3f}")
    
    # --- Корреляция ---
    t_emp, t_cr, significant = HypothesisTests.t_test_correlation(X, Y)
    r_value = stats_x.pearson_corr(Y)
    print(f"r = {r_value:.3f}, t_emp={t_emp:.3f}, t_cr={t_cr:.3f}, significant={significant}")
    
    # --- Регрессия ---
    reg = RegressionVerifier(X, Y)
    print(f"Regression: Y = {reg.a:.3f} + {reg.b:.3f}*X")
    print(f"R² = {reg.r_squared():.3f}")
    
    # --- F-test ---
    F_emp = reg.f_test()
    df_model = 1
    df_resid = len(X) - 2
    F_cr, F_significant = HypothesisTests.f_test(F_emp, df_model, df_resid)
    print(f"F-test: F_emp={F_emp:.3f}, F_cr={F_cr:.3f}, significant={F_significant}")
    
    # --- t-тесты ---
    t_a, t_b = reg.t_values()
    t_cr_a, sig_a = HypothesisTests.t_test(t_a, df_resid)
    t_cr_b, sig_b = HypothesisTests.t_test(t_b, df_resid)
    print(f"T-test a: t_emp={t_a:.3f}, t_cr={t_cr_a:.3f}, significant={sig_a}")
    print(f"T-test b: t_emp={t_b:.3f}, t_cr={t_cr_b:.3f}, significant={sig_b}")
    
    return reg, r_value, F_emp, F_cr, F_significant, t_a, sig_a, t_b, sig_b


# === 3. Сохранение отчёта ===
def save_report(reg, r_value, F_emp, F_cr, F_significant, t_a, sig_a, t_b, sig_b, label):
    report = f"""
==================================================
📊 Итоговый вывод по модели ({label})
==================================================
1. Корреляция:
   r = {r_value:.3f} → {'сильная' if abs(r_value) > 0.7 else 'слабая/умеренная'} зависимость

2. Уравнение регрессии:
   Y = {reg.a:.3f} + {reg.b:.3f} * X
   R² = {reg.r_squared():.3f} (объясняет {reg.r_squared()*100:.1f}% вариации индекса)

3. F-тест:
   F_emp={F_emp:.3f}, F_cr={F_cr:.3f} → значимость: {F_significant}

4. t-тесты:
   • константа a: t_emp={t_a:.3f}, значимость={sig_a}
   • коэффициент b: t_emp={t_b:.3f}, значимость={sig_b}

5. Практическая интерпретация:
   При росте ВВП на 1000 млрд руб. индекс MCFTR в среднем увеличивается на {reg.b*1000:.1f} пунктов.

⚠️ Ограничение:
Модель учитывает только ВВП. Для более точных прогнозов нужно добавить процентные ставки, инфляцию и внешние факторы.
==================================================
"""
    with open("report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print("\n📝 Отчёт сохранён в 'report.txt'")


# === 4. Запуск анализа ===
reg_model, r_value, F_emp, F_cr, F_significant, t_a, sig_a, t_b, sig_b = analyze_model(X, Y, "GDP → MCFTR Index")
save_report(reg_model, r_value, F_emp, F_cr, F_significant, t_a, sig_a, t_b, sig_b, "GDP → MCFTR Index")

# === 5. Визуализация ===
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color='blue', label='наблюдаемые значения')
plt.plot(X, reg_model.y_hat, color='red', linestyle='--', label='регрессия')
plt.xlabel("GDP (млрд руб.)")
plt.ylabel("MCFTR Index")
plt.title("Линейная регрессия: GDP → MCFTR Index")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("regression_plot.png", dpi=300, bbox_inches="tight")
plt.show()
print("📈 График сохранён как 'regression_plot.png'")