import os
import shutil
from sklearn.model_selection import train_test_split
import glob

def split_dataset(dataset_path, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Divide el dataset en train, val y test
    
    Args:
        dataset_path: Ruta al dataset (ej: 'data/detect/basketball-ball-3/')
        train_ratio: Proporción para entrenamiento (0.7 = 70%)
        val_ratio: Proporción para validación (0.2 = 20%)
        test_ratio: Proporción para prueba (0.1 = 10%)
    """
    
    # Verificar que las proporciones sumen 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Las proporciones deben sumar 1.0"
    
    # Rutas originales
    train_images_path = os.path.join(dataset_path, 'train', 'images')
    train_labels_path = os.path.join(dataset_path, 'train', 'labels')
    
    # Crear carpetas de destino
    for split in ['val', 'test']:
        os.makedirs(os.path.join(dataset_path, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, split, 'labels'), exist_ok=True)
    
    # Obtener lista de archivos de imagen
    image_files = glob.glob(os.path.join(train_images_path, '*'))
    image_names = [os.path.splitext(os.path.basename(f))[0] for f in image_files]
    
    print(f"Total de imágenes encontradas: {len(image_names)}")
    
    # Primera división: train vs (val + test)
    train_names, temp_names = train_test_split(
        image_names, 
        test_size=(val_ratio + test_ratio), 
        random_state=42
    )
    
    # Segunda división: val vs test
    val_names, test_names = train_test_split(
        temp_names, 
        test_size=test_ratio/(val_ratio + test_ratio), 
        random_state=42
    )
    
    print(f"Train: {len(train_names)} imágenes")
    print(f"Val: {len(val_names)} imágenes")
    print(f"Test: {len(test_names)} imágenes")
    
    # Función para mover archivos
    def move_files(names, split_name):
        for name in names:
            # Buscar archivo de imagen (puede tener diferentes extensiones)
            img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            img_moved = False
            
            for ext in img_extensions:
                src_img = os.path.join(train_images_path, name + ext)
                if os.path.exists(src_img):
                    dst_img = os.path.join(dataset_path, split_name, 'images', name + ext)
                    shutil.move(src_img, dst_img)
                    img_moved = True
                    break
            
            if not img_moved:
                print(f"Advertencia: No se encontró imagen para {name}")
            
            # Mover archivo de etiqueta
            src_label = os.path.join(train_labels_path, name + '.txt')
            if os.path.exists(src_label):
                dst_label = os.path.join(dataset_path, split_name, 'labels', name + '.txt')
                shutil.move(src_label, dst_label)
            else:
                print(f"Advertencia: No se encontró etiqueta para {name}")
    
    # Mover archivos a val y test (train se queda donde está)
    move_files(val_names, 'val')
    move_files(test_names, 'test')
    
    print(f"✅ División completada!")
    print(f"📁 {dataset_path}/train: {len(train_names)} archivos")
    print(f"📁 {dataset_path}/val: {len(val_names)} archivos")
    print(f"📁 {dataset_path}/test: {len(test_names)} archivos")

# Usar la función
split_dataset('./data/detect/basketball-ball-3/')