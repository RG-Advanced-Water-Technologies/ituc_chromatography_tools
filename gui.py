import tkinter as tk
from tkinter import filedialog
from calibration import HPLC_MS_Calibration

class CalibrationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HPLC-MS Calibration Tool")

        # Label und Button zum Auswählen der Excel-Datei
        self.label = tk.Label(root, text="Wähle Excel-Datei:")
        self.label.pack()

        self.select_button = tk.Button(root, text="Datei auswählen", command=self.load_excel_data)
        self.select_button.pack()

        # Textfeld zur Anzeige der Ergebnisse
        self.result_text = tk.Text(root, height=10, width=100)
        self.result_text.pack()

    def load_excel_data(self):
        # Öffnen eines Dateiauswahldialogs
        file_path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx")])

        if file_path:
            # Importieren der Daten aus der Excel-Datei
            calibration = HPLC_MS_Calibration.import_data_from_excel(file_path, "Tabelle1")

            # Durchführen der Kalibrierungsberechnungen
            calibration.fit_calibration_curve()
            LOD = calibration.calculate_LOD()
            LOQ = calibration.calculate_LOQ()
            S, RSD = calibration.calculate_regression_statistics()

            # Anzeigen der Ergebnisse im Textfeld
            results = f"LOD: {LOD}\nLOQ: {LOQ}\nStandardabweichung der Regression (S): {S}\nRelative Standardabweichung der Regression (RSD): {RSD}%"
            self.result_text.delete(1.0, tk.END)  # Löschen des vorhandenen Texts
            self.result_text.insert(tk.END, results)
            calibration.plot_calibration_curve()

if __name__ == "__main__":
    root = tk.Tk()
    app = CalibrationApp(root)
    root.mainloop()
