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



  

  

    
        
    
        
    
        
    


