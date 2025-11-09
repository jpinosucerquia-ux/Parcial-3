import os
from clases import GestorDICOM
import cv2

os.makedirs("resultados", exist_ok=True)

gestor = GestorDICOM()

while True:
    print("\n--- MENÚ PRINCIPAL ---")
    print("1. Cargar estudio DICOM")
    print("2. Mostrar cortes 3D")
    print("3. Aplicar zoom y recorte")
    print("4. Segmentar imagen")
    print("5. Transformación morfológica")
    print("6. Convertir a NIFTI")
    print("7. Mostrar cortes 3D con nilearn")
    print("8. Salir")

    opcion = input("Seleccione una opción: ")

    if opcion == '1':
        carpeta = input("Ingrese la ruta de la carpeta DICOM: ")
        estudio = gestor.cargar_estudio(carpeta)
        print("✅ Estudio cargado correctamente.")
    elif opcion == '2':
        if gestor.estudios:
            gestor.estudios[-1].mostrar_cortes()
        else:
            print("⚠️ No hay estudios cargados.")
    elif opcion == '3':
        x, y, w, h = map(int, input("Ingrese x, y, ancho, alto: ").split())
        nombre = input("Nombre para guardar el recorte: ")
        gestor.estudios[-1].zoom(x, y, w, h, nombre)
    elif opcion == '4':
        print("Tipos disponibles:\n0: BINARIO\n1: BINARIO_INV\n2: TRUNCADO\n3: TOZERO\n4: TOZERO_INV")
        tipo = int(input("Ingrese el tipo de binarización: "))
        tipos = [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV, cv2.THRESH_TRUNC, cv2.THRESH_TOZERO, cv2.THRESH_TOZERO_INV]
        gestor.estudios[-1].segmentar(tipos[tipo])
    elif opcion == '5':
        kernel = int(input("Ingrese tamaño de kernel: "))
        gestor.estudios[-1].transformacion_morfologica(kernel)
    elif opcion == '6':
        carpeta = input("Ingrese la carpeta DICOM a convertir: ")
        salida = "resultados/nifti"
        gestor.estudios[-1].convertir_a_nifti(carpeta, salida)
    elif opcion == '7':
        archivo = input("Ingrese la ruta del archivo NIFTI (.nii o .nii.gz): ")
        gestor.estudios[-1].mostrar_cortes_3d(archivo)
    elif opcion == '8':
        print(" Saliendo del programa...")
        break
    else:
        print(" Opción inválida.")
