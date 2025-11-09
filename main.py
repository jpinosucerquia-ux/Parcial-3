"""
main.py
Script principal con menú interactivo para usar las clases en clases.py
Requisitos implementados:
 - Permite cargar múltiples carpetas DICOM (puedes indicar varias rutas separadas por coma)
 - Reconstrucción 3D por carpeta y visualización de cortes (coronal, sagital, transversal)
 - Extracción de dataelements y guardado en CSV
 - Creación y gestión de objetos EstudioImaginologico
 - Operaciones: ZOOM (recorte+resize+dibujo), Segmentación, Transformación morfológica
 - Conversión a NIfTI
 - Guardado de objetos creados (pickle)

Instrucciones:
 1) Coloca este archivo en la misma carpeta que clases.py
 2) Ejecuta: python main.py
 3) Sigue el menú

Nota: Ajusta las rutas de 'datos/' y 'resultados/' según tu estructura.
"""

import os
import sys
from typing import List

from clases import GestorDICOM, EstudioImaginologico


RESULTS_DIR = 'resultados'
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR, exist_ok=True)


def seleccionar_carpeta_predeterminada() -> str:
    """Intenta listar carpetas dentro de ./datos y permitir selección rápida."""
    datos_dir = 'datos'
    if not os.path.exists(datos_dir):
        return ''
    folders = [f for f in os.listdir(datos_dir) if os.path.isdir(os.path.join(datos_dir, f))]
    if not folders:
        return ''
    print('\nCarpetas encontradas en ./datos:')
    for i, f in enumerate(folders, 1):
        print(f"  {i}. {f}")
    sel = input('Elija número de carpeta (Enter para cancelar): ')
    try:
        if sel.strip() == '':
            return ''
        idx = int(sel) - 1
        if 0 <= idx < len(folders):
            return os.path.join(datos_dir, folders[idx])
    except Exception:
        return ''
    return ''


def pedir_varias_rutas() -> List[str]:
    rutas = input('Ingrese rutas de carpetas DICOM separadas por coma (o deje vacío para elegir desde datos/): ')
    if rutas.strip() == '':
        auto = seleccionar_carpeta_predeterminada()
        if auto:
            return [auto]
        return []
    parts = [p.strip() for p in rutas.split(',') if p.strip()]
    return parts


def main():
    gestores: List[GestorDICOM] = []
    estudios: List[EstudioImaginologico] = []

    while True:
        print('\n===== MENÚ PRINCIPAL =====')
        print('1. Cargar carpeta(s) DICOM')
        print('2. Reconstruir 3D y mostrar cortes (elige gestor)')
        print('3. Extraer datos y guardar CSV (elige gestor)')
        print('4. Crear objeto EstudioImaginologico a partir de un gestor')
        print('5. Listar estudios creados')
        print('6. Zoom (recorte + resize + dibujar) sobre un estudio')
        print('7. Segmentación (umbral) sobre un estudio')
        print('8. Transformación morfológica sobre un estudio')
        print('9. Convertir volumen a NIfTI (elige gestor)')
        print('10. Guardar objetos creados (pickle)')
        print('11. Salir')

        opcion = input('Seleccione una opción: ').strip()

        if opcion == '1':
            rutas = pedir_varias_rutas()
            if not rutas:
                print('No se seleccionó ruta.')
                continue
            for r in rutas:
                try:
                    g = GestorDICOM(r)
                    print(f'Cargando {r} ...')
                    g.cargar_carpeta(r)
                    gestores.append(g)
                    print('Carga exitosa.')
                except Exception as e:
                    print(f'Error cargando {r}: {e}')

        elif opcion == '2':
            if not gestores:
                print('No hay gestores cargados. Primero cargue carpetas DICOM (opción 1).')
                continue
            for i, g in enumerate(gestores, 1):
                print(f'{i}. {getattr(g, "folder_path", "(sin ruta)")}')
            sel = input('Elija número de gestor: ')
            try:
                gi = int(sel) - 1
                gestor = gestores[gi]
                gestor.reconstruir_3D()
                gestor.mostrar_cortes_3D()
            except Exception as e:
                print('Selección inválida o error:', e)

        elif opcion == '3':
            if not gestores:
                print('No hay gestores cargados.')
                continue
            for i, g in enumerate(gestores, 1):
                print(f'{i}. {getattr(g, "folder_path", "(sin ruta)")}')
            sel = input('Elija número de gestor para extraer datos: ')
            try:
                gi = int(sel) - 1
                gestor = gestores[gi]
                df = gestor.extraer_datos()
                out = os.path.join(RESULTS_DIR, f'estudio_{gi+1}_datos.csv')
                gestor.guardar_csv(out)
                print(f'Datos guardados en {out}')
            except Exception as e:
                print('Error:', e)

        elif opcion == '4':
            if not gestores:
                print('No hay gestores cargados.')
                continue
            for i, g in enumerate(gestores, 1):
                print(f'{i}. {getattr(g, "folder_path", "(sin ruta)")}')
            sel = input('Elija número de gestor para crear EstudioImaginologico: ')
            try:
                gi = int(sel) - 1
                estudio = EstudioImaginologico(gestores[gi])
                estudios.append(estudio)
                print(f'Estudio creado: {estudio.nombre} (forma {estudio.forma})')
            except Exception as e:
                print('Error creando estudio:', e)

        elif opcion == '5':
            if not estudios:
                print('No hay estudios creados aún.')
            else:
                for i, e in enumerate(estudios, 1):
                    print(f'{i}. {getattr(e, "nombre", "(sin nombre)")} - StudyDate: {e.StudyDate} - Modality: {e.Modality} - forma: {e.forma}')

        elif opcion == '6':
            if not estudios:
                print('Cree primero un EstudioImaginologico (opción 4).')
                continue
            for i, e in enumerate(estudios, 1):
                print(f'{i}. {e.nombre} (forma {e.forma})')
            sel = input('Elija estudio: ')
            try:
                ei = int(sel) - 1
                est = estudios[ei]
                print('Ingrese rectángulo x y w h (ej: 50 40 100 120)')
                x, y, w, h = map(int, input().split())
                nombre_out = input('Nombre archivo salida (ej: resultados/recorte.png): ').strip()
                if not nombre_out:
                    nombre_out = os.path.join(RESULTS_DIR, f'recorte_estudio_{ei+1}.png')
                est.zoom(x, y, w, h, nombre_out)
                print('Recorte guardado en:', nombre_out)
            except Exception as e:
                print('Error en zoom:', e)

        elif opcion == '7':
            if not estudios:
                print('Cree primero un EstudioImaginologico (opción 4).')
                continue
            for i, e in enumerate(estudios, 1):
                print(f'{i}. {e.nombre} (forma {e.forma})')
            sel = input('Elija estudio: ')
            try:
                ei = int(sel) - 1
                est = estudios[ei]
                print('Tipos disponibles: binary, binary_inv, trunc, tozero, tozero_inv')
                tipo = input('Ingrese tipo de binarización: ').strip()
                thresh = input('Ingrese valor de umbral (Enter para Otsu): ').strip()
                threshv = int(thresh) if thresh else None
                res = est.segmentar(tipo, thresh_value=threshv)
                out = os.path.join(RESULTS_DIR, f'segmentacion_estudio_{ei+1}.png')
                # Guardar resultado
                import cv2
                cv2.imwrite(out, res)
                print('Segmentación guardada en', out)
            except Exception as e:
                print('Error en segmentación:', e)

        elif opcion == '8':
            if not estudios:
                print('Cree primero un EstudioImaginologico (opción 4).')
                continue
            for i, e in enumerate(estudios, 1):
                print(f'{i}. {e.nombre} (forma {e.forma})')
            sel = input('Elija estudio: ')
            try:
                ei = int(sel) - 1
                est = estudios[ei]
                print('Operaciones disponibles: open, close, gradient, tophat, blackhat')
                tipo = input('Ingrese tipo: ').strip()
                k = int(input('Ingrese tamaño de kernel (ej: 3): ').strip())
                res = est.transformacion_morfologica(tipo, k)
                out = os.path.join(RESULTS_DIR, f'morfologia_estudio_{ei+1}.png')
                import cv2
                cv2.imwrite(out, res)
                print('Resultado guardado en', out)
            except Exception as e:
                print('Error en morfología:', e)

        elif opcion == '9':
            if not gestores:
                print('No hay gestores cargados.')
                continue
            for i, g in enumerate(gestores, 1):
                print(f'{i}. {getattr(g, "folder_path", "(sin ruta)")}')
            sel = input('Elija número de gestor para convertir a NIfTI: ')
            try:
                gi = int(sel) - 1
                gestor = gestores[gi]
                out_folder = os.path.join(RESULTS_DIR, 'nifti_output')
                os.makedirs(out_folder, exist_ok=True)
                path = gestor.convertir_a_nifti(out_folder)
                print('NIfTI guardado en:', path)
            except Exception as e:
                print('Error:', e)

        elif opcion == '10':
            if not estudios:
                print('No hay objetos de estudio para guardar.')
                continue
            out = os.path.join(RESULTS_DIR, 'estudios_guardados.pkl')
            # Usar método del primer gestor para guardar (pickle)
            try:
                if gestores:
                    gestores[0].guardar_objetos(estudios, out)
                    print('Objetos guardados en', out)
                else:
                    # si no hay gestores, pickle directamente
                    import pickle
                    with open(out, 'wb') as f:
                        pickle.dump(estudios, f)
                    print('Objetos guardados en', out)
            except Exception as e:
                print('Error guardando objetos:', e)

        elif opcion == '11':
            print('Saliendo...')
            sys.exit(0)

        else:
            print('Opción inválida. Intente de nuevo.')


if __name__ == '__main__':
    main()

