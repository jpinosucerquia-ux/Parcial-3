import os
import cv2
import numpy as np
import pydicom
import dicom2nifti
import matplotlib.pyplot as plt
from nilearn import plotting


class GestorDICOM:
    def __init__(self):
        self.estudios = []

    def cargar_estudio(self, carpeta):
        if not os.path.exists(carpeta):
            print("Ruta no encontrada.")
            return None

        archivos = [f for f in os.listdir(carpeta) if f.endswith('.dcm')]
        if not archivos:
            print("No se encontraron archivos DICOM en la carpeta.")
            return None

        volumen = []
        for archivo in sorted(archivos):
            try:
                ds = pydicom.dcmread(os.path.join(carpeta, archivo))
                if hasattr(ds, "PixelData"):
                    volumen.append(ds.pixel_array)
            except Exception as e:
                print(f"Error leyendo {archivo}: {e}")

        if not volumen:
            print("No se pudieron leer imágenes DICOM válidas.")
            return None

        volumen_3d = np.stack(volumen, axis=0)
        ds0 = pydicom.dcmread(os.path.join(carpeta, archivos[0]))

        estudio = EstudioImaginologico(
            study_date=getattr(ds0, "StudyDate", "Desconocido"),
            study_time=getattr(ds0, "StudyTime", "Desconocido"),
            modality=getattr(ds0, "Modality", "Desconocida"),
            description=getattr(ds0, "StudyDescription", "Sin descripción"),
            series_time=getattr(ds0, "SeriesTime", "Desconocida"),
            imagen=volumen_3d,
            pixel_spacing=getattr(ds0, "PixelSpacing", [1.0, 1.0]),
            slice_thickness=float(getattr(ds0, "SliceThickness", 1.0))
        )
        self.estudios.append(estudio)
        return estudio


class EstudioImaginologico:
    def __init__(self, study_date, study_time, modality, description, series_time,
                 imagen, pixel_spacing, slice_thickness):
        self.study_date = study_date
        self.study_time = study_time
        self.modality = modality
        self.description = description
        self.series_time = series_time
        self.imagen = imagen
        self.pixel_spacing = pixel_spacing
        self.slice_thickness = slice_thickness

    def mostrar_cortes(self):
        z, y, x = self.imagen.shape
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(self.imagen[z // 2, :, :], cmap='gray')
        plt.title("Corte Transversal")

        plt.subplot(1, 3, 2)
        plt.imshow(self.imagen[:, y // 2, :], cmap='gray')
        plt.title("Corte Coronal")

        plt.subplot(1, 3, 3)
        plt.imshow(self.imagen[:, :, x // 2], cmap='gray')
        plt.title("Corte Sagital")

        plt.tight_layout()
        plt.show()

    def zoom(self, x, y, w, h, nombre_salida):
        img2d = self.imagen[self.imagen.shape[0] // 2, :, :]
        norm = ((img2d - np.min(img2d)) / (np.max(img2d) - np.min(img2d)) * 255).astype(np.uint8)
        color = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)

        cv2.rectangle(color, (x, y), (x + w, y + h), (0, 0, 255), 2)
        texto = f"{w * self.pixel_spacing[0]:.2f}mm x {h * self.pixel_spacing[1]:.2f}mm"
        cv2.putText(color, texto, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        recorte = norm[y:y + h, x:x + w]
        recorte_redimensionado = cv2.resize(recorte, (256, 256))

        plt.subplot(1, 2, 1)
        plt.imshow(color)
        plt.title("Imagen Original")

        plt.subplot(1, 2, 2)
        plt.imshow(recorte_redimensionado, cmap='gray')
        plt.title("Recorte y Redimensionado")

        plt.show()
        os.makedirs("resultados", exist_ok=True)
        cv2.imwrite(f"resultados/{nombre_salida}.png", recorte_redimensionado)

    def segmentar(self, tipo, nombre_salida):
        os.makedirs("resultados", exist_ok=True)
        img = self.imagen[self.imagen.shape[0] // 2, :, :]
        norm = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)
        _, segmented = cv2.threshold(norm, 127, 255, tipo)

        # Generate unique filename if it already exists
