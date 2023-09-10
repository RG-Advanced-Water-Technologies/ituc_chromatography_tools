import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

class CalibrationModel:
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        self.background_std = np.std(y_data[:2])
        self.popt = None

    def fit_calibration_curve(self, model_function):
        self.popt, _ = curve_fit(model_function, self.x_data, self.y_data)

    def plot_calibration_curve(self, xlabel, ylabel, title, model_function):
        # Create a figure with the calibration curve and residual plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [2, 0.8]})

        # Plot the calibration curve
        ax1.scatter(self.x_data, self.y_data, edgecolors='black', facecolors='white', label="Daten")
        ax1.plot(self.x_data, model_function(self.x_data, *self.popt), 'k--', label="Anpassung")
        #ax1.legend()
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.set_title(title)

        # Calculate residuals
        fitted_values = model_function(self.x_data, *self.popt)
        residuals = self.y_data - fitted_values

        # Plot the residual plot
        ax2.scatter(self.x_data, residuals, edgecolors='black', facecolors='gray', marker='o')
        ax2.axhline(y=0, color='gray', linestyle='-')
        #ax2.legend()
        ax2.set_xlabel("Standard")
        ax2.set_ylabel("Residual")

        # Calculate the convergence radius
        convergence_radius = np.max(np.abs(residuals))

        if convergence_radius is not None:
            # Add a circle representing the convergence radius
            circle = plt.Circle((self.x_data[-1], fitted_values[-1]), convergence_radius, color='g', fill=False, label='Convergence Radius')
            #ax1.add_artist(circle)

        # Show the combined figure
        plt.tight_layout()
        plt.show()

    def calculate_residuals(self,residuals):
        residuals = self.y_data - self.cubic_model(self.x_data, *self.popt)
        return residuals

    def calculate_LOD(self):
        slope = self.popt[1]
        LOD = 3 * (self.background_std / slope)
        return LOD

    def calculate_LOQ(self):
        slope = self.popt[1]
        LOQ = 10 * (self.background_std / slope)
        return LOQ


    @classmethod
    def import_data_from_excel(cls, excel_file, sheet_name):
        # Lese die Daten aus Excel-Datei
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        x_data = df["Konzentration"].values
        y_data = df["Signal"].values
        return cls(x_data, y_data)


class Exponential_Regression_Model(CalibrationModel):
    def exponential_model(self, x, A, B, C):
        return A * np.exp(-B * x) + C

    def fit_calibration_curve(self):
        super().fit_calibration_curve(self.exponential_model)

    def calculate_r_squared(self):
        self.fit_calibration_curve(self.exponential_model)
        fitted_values = self.linear_model(self.x_data, *self.popt)
        residuals = self.y_data - fitted_values
        RSS = np.sum(residuals**2)
        y_mean = np.mean(self.y_data)
        SST = np.sum((self.y_data - y_mean)**2)
        r_squared = 1 - (RSS / SST)
        return r_squared

class Cubic_Regression_Model(CalibrationModel):
    def cubic_model(self, x, A, B, C):
        return (-A*(x^2))+(-B*x) + C

    def fit_calibration_curve(self):
        super().fit_calibration_curve(self.cubic_model)

    def calculate_regression_statistics(self):
        S = np.std(residuals)
        mean_concentration = np.mean(self.x_data)
        RSD = (S / mean_concentration) * 100
        return S, RSD

    def calculate_r_squared(self):
        self.fit_calibration_curve(self.cubic_model)
        fitted_values = self.cubic_model(self.x_data, *self.popt)
        residuals = self.y_data - fitted_values
        RSS = np.sum(residuals**2)
        y_mean = np.mean(self.y_data)
        SST = np.sum((self.y_data - y_mean)**2)
        r_squared = 1 - (RSS / SST)
        return r_squared



class Linear_Regression_Model(CalibrationModel):
    def linear_model(self, x, A, B):
        return (-A * x) + B

    def fit_calibration_curve(self):
        super().fit_calibration_curve(self.linear_model)

    def calculate_regression_statistics(self):
        residuals = self.y_data - self.linear_model(self.x_data, *self.popt)
        S = np.std(residuals)
        mean_concentration = np.mean(self.x_data)
        RSD = (S / mean_concentration) * 100
        return S, RSD

    def calc_reg(self,residuals):
        S = np.std(residuals)
        mean_concentration = np.mean(self.x_data)
        RSD = (S / mean_concentration) * 100

        n = len(self.x_data)
        confidency = None  #TODO: Weitere Statistikparameter ergänzen!

        return S, RSD

    def calculate_r_squared(self):
        self.fit_calibration_curve(self.linear_model)
        fitted_values = self.linear_model(self.x_data, *self.popt)
        residuals = self.y_data - fitted_values
        RSS = np.sum(residuals**2)
        y_mean = np.mean(self.y_data)
        SST = np.sum((self.y_data - y_mean)**2)
        r_squared = 1 - (RSS / SST)
        return r_squared


if __name__ == "__main__":
    # Importiere Daten aus Excel-Datei
    excel_file = r"C:\Users\mi43qid\Documents\GitHub\numpic\ituc_chromatography_tools\data\test_nonlinear_calibration.xlsx"  # Passe den Dateinamen an
    sheet_name = "Tabelle1"  # Passe den Tabellennamen an

    calibration = Exponential_Model.import_data_from_excel(excel_file, sheet_name)
    calibration.fit_calibration_curve()

    LOD = calibration.calculate_LOD()
    LOQ = calibration.calculate_LOQ()

    print(f"LOD: {LOD}")
    print(f"LOQ: {LOQ}")

    calibration.plot_calibration_curve(xlabel="Konzentration [µg/L]", ylabel="Peakarea", title="Test der Kalibration", model_function=calibration.exponential_model)
