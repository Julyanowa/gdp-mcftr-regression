from collections import Counter

class StatsAnalyzer:
    def __init__(self, data):
        self.data = data
    
    def mean(self):
        return sum(self.data)/len(self.data)
    
    def variance(self):
         m = self.mean()
         n = len(self.data)
         return sum((x - m) ** 2 for x in self.data) / n
    
    def std_dev(self):
        return self.variance() ** 0.5
    
    def median(self):
        sorted_data = sorted(self.data)
        n = len(sorted_data)
        if n % 2 == 1:
            return sorted_data[n // 2]
        else:
            return (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
        
    def mode(self):
        #Возвращает список модальных значений (если их несколько)
        if not self.data:
            return None
        counts = Counter(self.data)
        max_count = max(counts.values())
        modes = [val for val, count in counts.items() if count == max_count]
        return modes    
    
    def variation(self):
        mean_copy = self.mean()           
        return self.std_dev() / mean_copy if mean_copy != 0 else None

    def skewness(self): #асимметрия
        n = len(self.data)
        m = self.mean()
        s = self.std_dev()
        if s == 0:
          return None
        return sum((x - m) ** 3 for x in self.data) / (n * s**3)
    
    def kurtosis(self): #эксцесс
        n = len(self.data)
        m = self.mean()
        s = self.std_dev()
        if s == 0:
            return None
        return sum((x - m) ** 4 for x in self.data) / (n * s**4) - 3
    
    def covariance(self, other_data):
        if len(self.data) != len(other_data):
         return None  # Длины выборок должны совпадать
        n = len(self.data)
        mean_x = self.mean()
        mean_y = sum(other_data) / n
        return sum((self.data[i] - mean_x) * (other_data[i] - mean_y) for i in range(n)) / n
    
    def pearson_corr(self, other_data):
        cov = self.covariance(other_data)
        std_x = self.std_dev()
        std_y = StatsAnalyzer(other_data).std_dev()
        if std_x == 0 or std_y == 0:
            return None
        return cov / (std_x * std_y)

if __name__ == "__main__":
    # Мои тестовые данные (X — выпуск, Y — затраты)
    X = [1, 2, 4, 3, 5, 3, 4]
    Y = [3, 7, 15, 10, 17, 10, 15]

    # Создаем объекты для анализа
    analyzer_X = StatsAnalyzer(X)
    analyzer_Y = StatsAnalyzer(Y)

    # Вычисления для X
    print("📊 Анализ выборки X (выпуск, тыс. шт.)")
    print(f"Среднее: {analyzer_X.mean():.2f}")
    print(f"Медиана: {analyzer_X.median():.2f}")
    print(f"Мода: {analyzer_X.mode()}")
    print(f"Дисперсия: {analyzer_X.variance():.2f}")
    print(f"Стандартное отклонение: {analyzer_X.std_dev():.2f}")
    print(f"Асимметрия: {analyzer_X.skewness():.2f}")

    # Вычисления для Y
    print("\n📊 Анализ выборки Y (затраты, усл. ед.)")
    print(f"Среднее: {analyzer_Y.mean():.2f}")
    print(f"Дисперсия: {analyzer_Y.variance():.2f}")
    print(f"Стандартное отклонение: {analyzer_Y.std_dev():.2f}")
    print(f"Асимметрия: {analyzer_Y.skewness():.2f}")

    # Совместный анализ (ковариация и корреляция)
    cov = analyzer_X.covariance(Y)
    corr = analyzer_X.pearson_corr(Y)

    print("\n📊 Совместный анализ X и Y")
    print(f"Ковариация: {cov:.2f}")
    print(f"Коэффициент корреляции Пирсона: {corr:.2f}")

  

  

    
        
    
        
    
        
    

