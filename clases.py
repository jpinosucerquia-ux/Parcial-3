import os
import cv2
import numpy as np
import pydicom
import dicom2nifti
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting

# -------------------- Clase 1: GestorDICOM --------------------

class GestorDICOM:
    def __init__(self):
        self.estudios = []

    def cargar_estudio(self, carpeta):
        """Carga todos los archivos DICOM de una carpeta y crea un EstudioImaginologico"""
        archivos = [f for f in os.listdir(carpeta) if f.endswith('.dcm')]
        volumen = []

        for archivo in sorted(archivos):
            ds = pydicom.dcmread(os.path.join(carpeta, archivo))
            volumen.append(ds.pixel_array)

        volumen_3d = np.stack(volumen, axis=0)
        ds0 = pydicom.dcmread(os.path.join(carpeta, archivos[0]))

        # Crear objeto EstudioImaginologico
        estudio = EstudioImaginologico(
            study_date=ds0.StudyDate,
            study_time=ds0.StudyTime,
            modality=ds0.Modality,
            description=getattr(ds0, "StudyDescription", "Sin descripción"),
            series_time=ds0.SeriesTime,
            imagen=volumen_3d,
            pixel_spacing=ds0.PixelSpacing,
            slice_thickness=float(ds0.SliceThickness)
        )
        self.estudios.append(estudio)
        return estudio

# -------------------- Clase 2: EstudioImaginologico --------------------

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

    # ---- Mostrar los 3 cortes (coronal, sagital, transversal)
    def mostrar_cortes(self):
        z, y, x = self.imagen.shape
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(self.imagen[z//2, :, :], cmap='gray')
        plt.title("Corte Transversal")

        plt.subplot(1, 3, 2)
        plt.imshow(self.imagen[:, y//2, :], cmap='gray')
        plt.title("Corte Coronal")

        plt.subplot(1, 3, 3)
        plt.imshow(self.imagen[:, :, x//2], cmap='gray')
        plt.title("Corte Sagital")

        plt.tight_layout()
        plt.show()

    # ---- Método de Zoom y recorte con OpenCV
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
        cv2.imwrite(f"resultados/{nombre_salida}.png", recorte_redimensionado)

    # ---- Segmentación
    def segmentar(self, tipo):
        img = self.imagen[self.imagen.shape[0] // 2, :, :]
        norm = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)
        _, segmented = cv2.threshold(norm, 127, 255, tipo)
        plt.imshow(segmented, cmap='gray')
        plt.title("Imagen Segmentada")
        plt.show()
        cv2.imwrite("resultados/segmentacion.png", segmented)

    # ---- Transformación morfológica
    def transformacion_morfologica(self, kernel_size):
        img = self.imagen[self.imagen.shape[0] // 2, :, :]
        norm = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        morf = cv2.morphologyEx(norm, cv2.MORPH_OPEN, kernel)
        plt.imshow(morf, cmap='gray')
        plt.title("Transformación Morfológica (Apertura)")
        plt.show()
        cv2.imwrite("resultados/morfologia.png", morf)

    # ---- Conversión a NIFTI usando dicom2nifti
    def convertir_a_nifti(self, carpeta_dicom, carpeta_salida):
        os.makedirs(carpeta_salida, exist_ok=True)
        dicom2nifti.convert_directory(carpeta_dicom, carpeta_salida)
        print(f"✅ Conversión a NIFTI completada y guardada en: {carpeta_salida}")

    # ---- Visualización 3D con nilearn
    def mostrar_cortes_3d(self, archivo_nifti):
        plotting.plot_anat(archivo_nifti, display_mode='ortho', title="Cortes 3D (NIfTI)")
        plotting.show()
