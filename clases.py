"""
clases.py
Plantilla de clases para el proyecto P3: gestión y procesamiento de DICOM.
Contiene:
 - GestorDICOM: carga carpetas DICOM, reconstruye volumen 3D, extrae dataelements, guarda CSV, convierte a NIfTI.
 - EstudioImaginologico: objeto que representa un estudio (atributos obligatorios) y funciones de zoom, segmentación y transformaciones morfológicas.

Dependencias permitidas: pydicom, numpy, pandas, matplotlib, opencv (cv2), nibabel, glob, os, datetime, pickle

Nota: Este archivo es una plantilla funcional y contiene implementación básica. Ajusta rutas y prueba con tus carpetas de `datos/`.
"""

import os
import glob
import pickle
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
import cv2
import nibabel as nib


class GestorDICOM:
    """Clase para gestionar carpetas DICOM y reconstruir volúmenes 3D.

    Uso típico:
        gestor = GestorDICOM('/ruta/a/carpeta')
        gestor.cargar_carpeta()
        gestor.reconstruir_3D()
        gestor.mostrar_cortes_3D()
        df = gestor.extraer_datos()
        gestor.guardar_csv('resultados/estudio.csv')
        gestor.convertir_a_nifti('resultados/nifti_output')
    """

    def __init__(self, folder_path: Optional[str] = None):
        self.folder_path = folder_path
        self.datasets: List[pydicom.dataset.FileDataset] = []
        self.image_3d: Optional[np.ndarray] = None
        self.pixel_spacing: Optional[Tuple[float, float]] = None
        self.slice_thickness: Optional[float] = None

    def cargar_carpeta(self, folder_path: Optional[str] = None) -> None:
        """Carga todos los archivos DICOM de una carpeta (no recursiva por defecto).

        Guarda la lista de datasets en self.datasets.
        """
        if folder_path is not None:
            self.folder_path = folder_path
        if self.folder_path is None:
            raise ValueError("No se ha proporcionado folder_path")

        files = sorted(glob.glob(os.path.join(self.folder_path, '*')))
        datasets = []
        for f in files:
            try:
                ds = pydicom.dcmread(f, force=True)
                # omit non-image files gracefully
                if hasattr(ds, 'PixelData'):
                    datasets.append(ds)
            except Exception:
                # ignorar archivos que no son DICOM
                continue

        if not datasets:
            raise FileNotFoundError(f"No se encontraron archivos DICOM con PixelData en {self.folder_path}")

        self.datasets = datasets

        # Obtener pixel spacing y slice thickness del primer dataset válido
        first = datasets[0]
        if hasattr(first, 'PixelSpacing'):
            ps = first.PixelSpacing
            # PixelSpacing en pydicom suele ser [row_spacing, col_spacing]
            self.pixel_spacing = (float(ps[0]), float(ps[1]))
        if hasattr(first, 'SliceThickness'):
            self.slice_thickness = float(first.SliceThickness)

    def reconstruir_3D(self) -> None:
        """Reconstruye una matriz 3D apilando cortes ordenados por ImagePositionPatient (z).

        Si ImagePositionPatient no existe, intenta usar InstanceNumber.
        """
        if not self.datasets:
            raise RuntimeError("No hay datasets cargados. Ejecute cargar_carpeta().")

        # Intentar ordenar por ImagePositionPatient[2]
        def _zpos(ds):
            if hasattr(ds, 'ImagePositionPatient'):
                return float(ds.ImagePositionPatient[2])
            if hasattr(ds, 'SliceLocation'):
                return float(ds.SliceLocation)
            if hasattr(ds, 'InstanceNumber'):
                return float(ds.InstanceNumber)
            return 0.0

        slices = sorted(self.datasets, key=_zpos)
        arrays = [s.pixel_array.astype(np.float32) for s in slices]

        # Apilar en eje z
        volume = np.stack(arrays, axis=0)
        self.image_3d = volume

        # Si aún no tenemos pixel spacing y slice_thickness, intentar extraerlos
        if self.pixel_spacing is None and hasattr(slices[0], 'PixelSpacing'):
            ps = slices[0].PixelSpacing
            self.pixel_spacing = (float(ps[0]), float(ps[1]))
        if self.slice_thickness is None and hasattr(slices[0], 'SliceThickness'):
            self.slice_thickness = float(slices[0].SliceThickness)

    def mostrar_cortes_3D(self, cmap: str = 'gray') -> None:
        """Muestra los cortes coronal, sagital y transversal en 3 subplots.

        - transversal: slice medio en eje z (axial)
        - coronal: slice medio en eje y
        - sagital: slice medio en eje x
        """
        if self.image_3d is None:
            raise RuntimeError("Volumen 3D no reconstruido. Ejecute reconstruir_3D().")

        vol = self.image_3d
        zc = vol.shape[0] // 2
        yc = vol.shape[1] // 2
        xc = vol.shape[2] // 2

        axial = vol[zc, :, :]
        coronal = vol[:, yc, :]
        sagittal = vol[:, :, xc]

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(coronal.T, cmap=cmap, origin='lower')
        axs[0].set_title('Coronal')
        axs[1].imshow(sagittal.T, cmap=cmap, origin='lower')
        axs[1].set_title('Sagital')
        axs[2].imshow(axial.T, cmap=cmap, origin='lower')
        axs[2].set_title('Transversal')
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    def extraer_datos(self) -> pd.DataFrame:
        """Extrae los dataelements relevantes de los datasets y retorna un DataFrame.

        Extrae: StudyDate, StudyTime, Modality, StudyDescription, SeriesTime, SeriesInstanceUID, PatientID
        """
        records = []
        for ds in self.datasets:
            rec = {
                'StudyDate': getattr(ds, 'StudyDate', ''),
                'StudyTime': getattr(ds, 'StudyTime', ''),
                'Modality': getattr(ds, 'Modality', ''),
                'StudyDescription': getattr(ds, 'StudyDescription', ''),
                'SeriesTime': getattr(ds, 'SeriesTime', ''),
                'SeriesInstanceUID': getattr(ds, 'SeriesInstanceUID', ''),
                'SOPInstanceUID': getattr(ds, 'SOPInstanceUID', ''),
                'PatientID': getattr(ds, 'PatientID', ''),
            }
            records.append(rec)
        df = pd.DataFrame(records)
        return df

    def guardar_csv(self, out_path: str) -> None:
        """Guarda el DataFrame (extraer_datos) en CSV.

        Si el directorio no existe, lo crea.
        """
        df = self.extraer_datos()
        dirn = os.path.dirname(out_path)
        if dirn and not os.path.exists(dirn):
            os.makedirs(dirn, exist_ok=True)
        df.to_csv(out_path, index=False)

    def convertir_a_nifti(self, output_folder: str) -> str:
        """Convierte el volumen 3D reconstruido a NIfTI y guarda el archivo.

        Retorna la ruta al archivo NIfTI.
        """
        if self.image_3d is None:
            raise RuntimeError("Volumen 3D no reconstruido. Ejecute reconstruir_3D().")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)

        # Affine básico (identidad). Para uso real se debe construir a partir de ImageOrientationPatient y PixelSpacing
        affine = np.eye(4)
        nifti_img = nib.Nifti1Image(self.image_3d.astype(np.float32), affine)
        out_file = os.path.join(output_folder, 'volume.nii.gz')
        nib.save(nifti_img, out_file)
        return out_file

    def guardar_objetos(self, objetos: List[object], out_file: str) -> None:
        """Guarda una lista de objetos python (por ejemplo EstudioImaginologico) usando pickle.

        Esto cumple el requisito simple de "almacenar los objetos creados".
        """
        dirn = os.path.dirname(out_file)
        if dirn and not os.path.exists(dirn):
            os.makedirs(dirn, exist_ok=True)
        with open(out_file, 'wb') as f:
            pickle.dump(objetos, f)


class EstudioImaginologico:
    """Representa un estudio imagénologico.

    Atributos obligatorios:
     - StudyDate, StudyTime, Modality, StudyDescription, SeriesTime, duracion, imagen (matriz 3D), forma

    Métodos principales:
     - zoom(x,y,w,h, salida, slice_index=None)
     - segmentar(tipo, slice_index=None, thresh_value=None)
     - transformacion_morfologica(tipo, kernel_size, slice_index=None)
    """

    def __init__(self, gestor: GestorDICOM):
        if gestor.image_3d is None:
            raise ValueError('El GestorDICOM no tiene volumen 3D. Ejecute reconstruir_3D().')

        # Tomar datos representativos del primer dataset
        ds0 = gestor.datasets[0]
        self.StudyDate = getattr(ds0, 'StudyDate', '')
        self.StudyTime = getattr(ds0, 'StudyTime', '')
        self.Modality = getattr(ds0, 'Modality', '')
        self.StudyDescription = getattr(ds0, 'StudyDescription', '')
        self.SeriesTime = getattr(ds0, 'SeriesTime', '')

        # Calcular duracion si es posible
        self.duracion = self._calcular_duracion(self.StudyTime, self.SeriesTime)

        self.imagen_3D = gestor.image_3d
        self.forma = self.imagen_3D.shape

        # Espaciado físico
        self.pixel_spacing = gestor.pixel_spacing
        self.slice_thickness = gestor.slice_thickness

        # Nombre por defecto
        self.nombre = f"Estudio_{self.StudyDate}_{self.Modality}"

    def _calcular_duracion(self, study_time: str, series_time: str) -> Optional[float]:
        try:
            # pydicom suele dar times como HHMMSS.frac o HHMMSS
            t1 = datetime.strptime(study_time.split('.')[0], '%H%M%S')
            t2 = datetime.strptime(series_time.split('.')[0], '%H%M%S')
            dur = (t2 - t1).total_seconds()
            return dur
        except Exception:
            return None

    @staticmethod
    def _normalizar_a_uint8(img: np.ndarray) -> np.ndarray:
        """Normaliza la imagen a rango 0-255 y la convierte a uint8.

        Fórmula: (img - min) / (max - min) * 255
        """
        arr = img.astype(np.float32)
        mn = np.nanmin(arr)
        mx = np.nanmax(arr)
        if mx == mn:
            res = np.zeros_like(arr, dtype=np.uint8)
            return res
        norm = (arr - mn) / (mx - mn) * 255.0
        return np.clip(norm, 0, 255).astype(np.uint8)

    def zoom(self, x: int, y: int, w: int, h: int, nombre_salida: str, slice_index: Optional[int] = None) -> None:
        """Recorta un rectángulo de la matriz 3D (slice indicado o slice medio) y lo procesa con OpenCV.

        - Convierte a uint8
        - Convierte a BGR para dibujar el rectángulo en color
        - Muestra la imagen original y la recortada (después del resize)
        - Guarda la imagen recortada en disco con el nombre dado

        x,y,w,h están en coordenadas de píxeles sobre la imagen 2D seleccionada.
        """
        if slice_index is None:
            slice_index = self.imagen_3D.shape[0] // 2

        img2d = self.imagen_3D[slice_index, :, :]
        img_u8 = self._normalizar_a_uint8(img2d)

        # Seleccionar ROI y convertir a BGR
        h_img, w_img = img_u8.shape
        x2 = max(0, x)
        y2 = max(0, y)
        xend = min(w_img, x + w)
        yend = min(h_img, y + h)

        roi = img_u8[y2:yend, x2:xend]
        roi_bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

        # Dibujar rectángulo sobre una copia de la imagen original (convertida a BGR)
        img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(img_bgr, (x2, y2), (xend, yend), (0, 255, 0), 2)

        # Calcular dimensiones en mm si pixel_spacing está presente
        if self.pixel_spacing is not None:
            row_sp, col_sp = self.pixel_spacing
            width_mm = (xend - x2) * col_sp
            height_mm = (yend - y2) * row_sp
            dim_text = f"{width_mm:.1f}mm x {height_mm:.1f}mm"
        else:
            dim_text = "N/A mm"

        # Escribir texto en la imagen del rectángulo
        cv2.putText(img_bgr, dim_text, (x2, max(15, y2 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Redimensionar el ROI a un tamaño razonable para visualización
        if roi_bgr.size == 0:
            raise ValueError('ROI vacío. Verifique coordenadas y dimensiones.')
        resized = cv2.resize(roi_bgr, (min(512, roi_bgr.shape[1]), min(512, roi_bgr.shape[0])), interpolation=cv2.INTER_AREA)

        # Mostrar original vs recorte
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        axs[0].set_title('Original con rectángulo')
        axs[0].axis('off')
        axs[1].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        axs[1].set_title('Recorte redimensionado')
        axs[1].axis('off')
        plt.tight_layout()
        plt.show()

        # Guardar el recorte en uint8 gris (o BGR según prefieras)
        out_dir = os.path.dirname(nombre_salida)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        # Guardar el recorte original normalizado como PNG
        cv2.imwrite(nombre_salida, roi_bgr)

    def segmentar(self, tipo: str = 'binary', slice_index: Optional[int] = None, thresh_value: Optional[int] = None) -> np.ndarray:
        """Segmenta un slice usando cv2.threshold. Retorna la imagen binaria resultante (uint8).

        tipo: 'binary', 'binary_inv', 'trunc', 'tozero', 'tozero_inv'
        thresh_value: si None, se calcula por Otsu
        """
        if slice_index is None:
            slice_index = self.imagen_3D.shape[0] // 2
        img2d = self.imagen_3D[slice_index, :, :]
        img_u8 = self._normalizar_a_uint8(img2d)

        type_map = {
            'binary': cv2.THRESH_BINARY,
            'binary_inv': cv2.THRESH_BINARY_INV,
            'trunc': cv2.THRESH_TRUNC,
            'tozero': cv2.THRESH_TOZERO,
            'tozero_inv': cv2.THRESH_TOZERO_INV,
        }
        if tipo not in type_map:
            raise ValueError(f"Tipo desconocido: {tipo}. Opciones: {list(type_map.keys())}")

        if thresh_value is None:
            # Otsu
            _, dst = cv2.threshold(img_u8, 0, 255, type_map[tipo] | cv2.THRESH_OTSU)
        else:
            _, dst = cv2.threshold(img_u8, thresh_value, 255, type_map[tipo])

        # Mostrar resultado
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(img_u8, cmap='gray')
        axs[0].set_title('Original (uint8)')
        axs[0].axis('off')
        axs[1].imshow(dst, cmap='gray')
        axs[1].set_title(f'Segmentación: {tipo}')
        axs[1].axis('off')
        plt.tight_layout()
        plt.show()

        return dst

    def transformacion_morfologica(self, tipo: str = 'open', kernel_size: int = 3, slice_index: Optional[int] = None) -> np.ndarray:
        """Aplica una transformación morfológica a un slice segmentado o al uint8 normalizado.

        tipo: 'open', 'close', 'gradient', 'tophat', 'blackhat'
        """
        if slice_index is None:
            slice_index = self.imagen_3D.shape[0] // 2
        img2d = self.imagen_3D[slice_index, :, :]
        img_u8 = self._normalizar_a_uint8(img2d)

        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        op_map = {
            'open': cv2.MORPH_OPEN,
            'close': cv2.MORPH_CLOSE,
            'gradient': cv2.MORPH_GRADIENT,
            'tophat': cv2.MORPH_TOPHAT,
            'blackhat': cv2.MORPH_BLACKHAT,
        }
        if tipo not in op_map:
            raise ValueError(f"Tipo desconocido: {tipo}. Opciones: {list(op_map.keys())}")

        res = cv2.morphologyEx(img_u8, op_map[tipo], kernel)

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(img_u8, cmap='gray')
        axs[0].set_title('Original (uint8)')
        axs[0].axis('off')
        axs[1].imshow(res, cmap='gray')
        axs[1].set_title(f'Morfología: {tipo}, k={kernel_size}')
        axs[1].axis('off')
        plt.tight_layout()
        plt.show()

        return res


# Helper para pruebas rápidas
if __name__ == '__main__':
    print("Este archivo contiene las clases GestorDICOM y EstudioImaginologico. Importa y úsalas desde main.py")
