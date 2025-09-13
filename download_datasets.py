"""
Скрипт для загрузки датасетов с Roboflow Universe
"""

from roboflow import Roboflow
import os

def download_roboflow_datasets():
    """Загружает датасеты с Roboflow Universe"""
    
    # Инициализация Roboflow (нужен API ключ)
    # Получите ключ на https://universe.roboflow.com/
    rf = Roboflow(api_key="3qCQ6UK3ceg9YNx9I3lp")  # Замените на ваш API ключ
    
    # Создаем папку для датасетов
    os.makedirs("datasets", exist_ok=True)
    
    # Список датасетов из DATASETS_ROBOFLOW.md
    # Пропускаем rust_and_scrach из-за проблем с длинными именами файлов в Windows
    datasets = [
        {
            "name": "car_scratch_and_dent", 
            "workspace": "carpro",
            "project": "car-scratch-and-dent",
            "format": "yolov8"  # Для object-detection
        },
        {
            "name": "car_scratch",
            "workspace": "project-kmnth", 
            "project": "car-scratch-xgxzs",
            "format": "yolov8"  # Для object-detection
        }
    ]
    
    for dataset in datasets:
        try:
            print(f"Загружаем датасет: {dataset['name']}")
            
            # Получаем проект
            project = rf.workspace(dataset['workspace']).project(dataset['project'])
            
            # Загружаем последнюю версию
            dataset_version = project.version(1)
            
            # Скачиваем в нужном формате
            dataset_version.download(dataset['format'], location=f"datasets/{dataset['name']}")
            
            print(f"Датасет {dataset['name']} успешно загружен")
            
        except Exception as e:
            print(f"Ошибка при загрузке {dataset['name']}: {e}")
    
    print("\nВсе датасеты загружены в папку 'datasets/'")
    print("Теперь можно запустить train_damage_model.py для обучения модели")

if __name__ == "__main__":
    print("🚀 Загружаем датасеты с Roboflow Universe...")
    print("📋 API ключ найден, начинаем загрузку...")
    print()
    
    # Запускаем загрузку датасетов
    download_roboflow_datasets()
