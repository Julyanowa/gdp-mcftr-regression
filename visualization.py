import matplotlib.pyplot as plt
from model_verification import RegressionVerifier  # импорт твоего класса


class RegressionVisualizer:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.model = RegressionVerifier(x, y)
        self.y_hat = self.model.y_hat
        self.residuals = self.model.residuals

    def plot_regression(self):
        """График регрессии с остатками"""
        plt.figure(figsize=(8, 6))

        # точки
        plt.scatter(self.x, self.y, color="blue", label="Наблюдаемые значения")

        # линия регрессии
        plt.plot(self.x, self.y_hat, color="red", linewidth=2,
                 label=f"Линия регрессии: y = {self.model.a:.2f} + {self.model.b:.2f}x")

        # остатки (вертикальные отрезки)
        for xi, yi, yhi in zip(self.x, self.y, self.y_hat):
            plt.vlines(xi, ymin=min(yi, yhi), ymax=max(yi, yhi), color="gray", linestyle="dotted")

        plt.title("Линейная регрессия")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    def plot_residuals(self):
        """График остатков"""
        plt.figure(figsize=(8, 5))
        plt.scatter(self.x, self.residuals, color="purple")
        plt.axhline(0, color="black", linestyle="--")
        plt.title("График остатков")
        plt.xlabel("X")
        plt.ylabel("Остатки")
        plt.grid(alpha=0.3)
        plt.show()


