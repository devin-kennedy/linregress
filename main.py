import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from tqdm import tqdm


class LinearRegression:
    def __init__(self, xs, ys):

        if xs is None:
            xs = np.zeros(10)

        if ys is None:
            ys = np.zeros(10)

        self.xs = xs
        self.ys = ys
        self.slope = 0
        self.intercept = 0

    def loss(self, slope=None, b=None):
        if not slope:
            slope = self.slope
        if not b:
            b = self.intercept

        # Original loss funtion:
        # return np.sum((slope * self.xs - self.ys) ** 2)
        return np.sum((slope * self.xs + b - self.ys) ** 2)

    def fit(self, delta=(2 ** -10), gamma=(2 ** -20)):
        gradient = None
        it = 0
        headers = ["Iteration", "Slope", "Gradient", "Loss", "Improved", "Intercept"]
        table = []
        while gradient != 0.0:
            gradient = (
                    (self.loss(self.slope + delta) - self.loss(self.slope - delta))
                    / (2 * delta)
            )
            oldSlope = self.slope
            self.slope -= gamma * gradient
            improved = self.loss(self.slope) < self.loss(oldSlope)
            table.append([it, self.slope, gradient, self.loss(self.slope), str(improved), None])
            it += 1
        print(tabulate(table, headers=headers))
        print(f"Completed with a slope of {self.slope} after {it} iterations")

    def fit2D(self, delta=(2 ** -10), gamma=(1**-5), epochs=(10**6)):
        headers = ["Iteration",
                   "Slope",
                   "Intercept", "Gradient (Fixed intercept)", "Gradient (Fixed slope)", "Loss", "Improved", "Gamma"]
        table = []
        for i in tqdm(range(epochs)):
            gradientIFixed = (
                (self.loss(self.slope + delta, self.intercept) - self.loss(self.slope - delta, self.intercept))
                / (2 * delta)
            )
            gradientSFixed = (
                (self.loss(self.slope, self.intercept + delta) - self.loss(self.slope, self.intercept - delta))
                / (2 * delta)
            )

            new_slope = self.slope - gamma * gradientIFixed
            new_intercept = self.intercept - gamma * gradientSFixed
            improved = self.loss(new_slope, new_intercept) < self.loss(self.slope, self.intercept)

            if improved:
                self.slope = new_slope
                self.intercept = new_intercept
            else:
                gamma /= 2

            table.append([i,
                          self.slope,
                          self.intercept, gradientIFixed, gradientSFixed, self.loss(), str(improved), gamma])
        print("Tabulating results...")
        print(tabulate(table, headers=headers))
        print(f"\nFinal slope: {self.slope}\nFinal intercept: {self.intercept}\nFinal gamma: {gamma}")

    def predict(self, xs):
        return xs * self.slope


def main():
    # Bill length is xs, flipper length is ys
    adelie_bill_len_mm = np.loadtxt("adelie.csv", delimiter=',', skiprows=1, usecols=0)
    adelie_flipper_len_mm = np.loadtxt("adelie.csv", delimiter=',', skiprows=1, usecols=1)

    regress = LinearRegression(adelie_bill_len_mm, adelie_flipper_len_mm)
    regress.fit2D(
        epochs=1500000
    )


if __name__ == '__main__':
    main()
