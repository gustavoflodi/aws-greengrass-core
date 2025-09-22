#!/usr/bin/env python3
import json
import time
import logging
import sys
import traceback
import os
import random
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

try:
    # Imports específicos do Greengrass IPC
    import awsiot.greengrasscoreipc
    from awsiot.greengrasscoreipc.model import (
        PublishToIoTCoreRequest,
        QOS
    )
    logger.info("✅ Imports do Greengrass IPC realizados com sucesso")
except ImportError as e:
    logger.error(f"❌ Erro ao importar módulos do Greengrass: {e}")
    sys.exit(1)

class GreengrassCarCounter:
    def __init__(self):
        self.ipc_client = None
        self.component_name = "com.example.CarCounterGreengrass"
        self.topic = "greengrass/v2/car-counter/results"
        
        # Configurações do contador de carros
        self.base_dir = "/greengrass/v2/packages/artifacts/com.example.CarCounterGreengrass/1.0.6/data/images"
        self.results_dir = "/greengrass/v2/packages/artifacts/com.example.CarCounterGreengrass/1.0.6/data/results"
        self.model = None
        
        # ✅ Configurar cache do PyTorch em diretório com permissão
        self.cache_dir = "/tmp/pytorch_cache"
        self.setup_cache_directory()
        
        # Classes de veículos que vamos contar (IDs do COCO dataset)
        # 2: car, 3: motorcycle, 5: bus, 7: truck
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle', 
            5: 'bus',
            7: 'truck'
        }
        
    def setup_cache_directory(self):
        """Configura diretório de cache do PyTorch com permissões corretas"""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            os.makedirs(self.results_dir, exist_ok=True)
            
            # Configurar variáveis de ambiente para PyTorch usar nosso cache
            os.environ['TORCH_HOME'] = self.cache_dir
            os.environ['TORCH_CACHE'] = self.cache_dir
            
            logger.info(f"✅ Diretórios configurados - Cache: {self.cache_dir}, Resultados: {self.results_dir}")
        except Exception as e:
            logger.error(f"❌ Erro ao configurar diretórios: {e}")
            # Tentar usar /tmp como fallback
            self.cache_dir = "/tmp"
            self.results_dir = "/tmp/results"
            os.environ['TORCH_HOME'] = self.cache_dir
            logger.info(f"⚠️ Usando /tmp como diretório de cache: {self.cache_dir}")
        
    def connect_ipc(self):
        """Estabelece conexão IPC com o Greengrass Core"""
        try:
            logger.info("🔄 Tentando estabelecer conexão IPC...")
            
            self.ipc_client = awsiot.greengrasscoreipc.connect()
            
            logger.info("✅ Conexão IPC estabelecida com sucesso!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao estabelecer conexão IPC: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def load_yolo_model(self):
        """Carrega o modelo YOLOv5 para detecção de objetos"""
        try:
            logger.info("🔄 Carregando modelo YOLOv5 para detecção de carros...")
            logger.info(f"📁 Cache directory: {os.environ.get('TORCH_HOME', 'default')}")
            
            # Carregar YOLOv5s (versão pequena e rápida)
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.model.eval()
            
            # Configurar confiança mínima
            self.model.conf = 0.5  # Confiança mínima de 50%
            self.model.iou = 0.45   # IoU threshold para NMS
            
            logger.info("✅ Modelo YOLOv5 carregado com sucesso!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo YOLO: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Fallback: tentar carregar modelo local se disponível
            try:
                logger.info("🔄 Tentando carregar modelo YOLOv5 local...")
                self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=False)
                logger.warning("⚠️ Modelo carregado sem pesos pré-treinados - detecções não funcionarão corretamente")
                return True
            except Exception as e2:
                logger.error(f"❌ Erro também no fallback: {e2}")
                return False
    
    def detect_vehicles(self, image_path):
        """Detecta veículos em uma imagem usando YOLOv5"""
        try:
            # Carregar imagem
            img = Image.open(image_path).convert("RGB")
            
            # Fazer detecção
            results = self.model(img)
            
            # Processar resultados
            detections = results.pandas().xyxy[0]  # Resultados em formato pandas
            
            # Filtrar apenas veículos
            vehicle_detections = detections[detections['class'].isin(self.vehicle_classes.keys())]
            
            # Contar por tipo de veículo
            vehicle_counts = {}
            total_vehicles = 0
            
            for class_id in self.vehicle_classes.keys():
                count = len(vehicle_detections[vehicle_detections['class'] == class_id])
                vehicle_name = self.vehicle_classes[class_id]
                vehicle_counts[vehicle_name] = count
                total_vehicles += count
            
            # Criar imagem anotada (opcional, para debug)
            annotated_img = self.annotate_image(img, vehicle_detections)
            
            return {
                'total_vehicles': total_vehicles,
                'vehicle_counts': vehicle_counts,
                'detections': len(vehicle_detections),
                'confidence_avg': float(vehicle_detections['confidence'].mean()) if len(vehicle_detections) > 0 else 0.0,
                'annotated_image': annotated_img
            }
            
        except Exception as e:
            logger.error(f"❌ Erro ao detectar veículos em {image_path}: {e}")
            return None
    
    def annotate_image(self, img, detections):
        """Anota a imagem com as detecções encontradas"""
        try:
            # Converter PIL para OpenCV
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # Desenhar bounding boxes
            for _, detection in detections.iterrows():
                x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
                class_id = int(detection['class'])
                confidence = detection['confidence']
                
                # Cor baseada no tipo de veículo
                color = (0, 255, 0) if class_id == 2 else (255, 0, 0)  # Verde para carros, vermelho para outros
                
                # Desenhar retângulo
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
                
                # Texto da label
                label = f"{self.vehicle_classes.get(class_id, 'vehicle')} {confidence:.2f}"
                cv2.putText(img_cv, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Converter de volta para PIL
            annotated_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            return annotated_pil
            
        except Exception as e:
            logger.error(f"❌ Erro ao anotar imagem: {e}")
            return img
    
    def save_annotated_image(self, annotated_img, original_filename):
        """Salva a imagem anotada no diretório de resultados"""
        try:
            if annotated_img:
                result_filename = f"annotated_{original_filename}"
                result_path = os.path.join(self.results_dir, result_filename)
                annotated_img.save(result_path)
                logger.info(f"💾 Imagem anotada salva: {result_path}")
                return result_filename
        except Exception as e:
            logger.error(f"❌ Erro ao salvar imagem anotada: {e}")
        return None
    
    def get_random_image(self):
        """Seleciona uma imagem aleatória do diretório"""
        try:
            images_dir = Path(self.base_dir)
            
            if not images_dir.exists():
                logger.error(f"❌ Diretório {self.base_dir} não existe")
                return None
            
            # Buscar imagens
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
            image_files = []
            
            for extension in image_extensions:
                image_files.extend(images_dir.glob(extension))
            
            if not image_files:
                logger.warning(f"⚠️ Nenhuma imagem encontrada em {self.base_dir}")
                return None
            
            # Selecionar aleatoriamente
            selected_image = random.choice(image_files)
            logger.info(f"🎲 Imagem selecionada aleatoriamente: {selected_image.name}")
            return selected_image
            
        except Exception as e:
            logger.error(f"❌ Erro ao selecionar imagem aleatória: {e}")
            return None
    
    def publish_car_count(self, image_name, detection_result):
        """Publica resultado da contagem de carros via IPC"""
        try:
            if not self.ipc_client:
                logger.error("❌ Cliente IPC não está conectado")
                return False

            message_data = {
                "timestamp": datetime.now().isoformat(),
                "component": self.component_name,
                "image_name": image_name,
                "vehicle_detection": {
                    "total_vehicles": detection_result['total_vehicles'],
                    "cars": detection_result['vehicle_counts'].get('car', 0),
                    "motorcycles": detection_result['vehicle_counts'].get('motorcycle', 0),
                    "buses": detection_result['vehicle_counts'].get('bus', 0),
                    "trucks": detection_result['vehicle_counts'].get('truck', 0),
                    "confidence_average": detection_result['confidence_avg']
                },
                "thing_name": "MeuCoreWSLDockerV2"
            }

            message_json = json.dumps(message_data, indent=2)

            publish_request = PublishToIoTCoreRequest(
                topic_name=self.topic,
                qos=QOS.AT_LEAST_ONCE,
                payload=bytes(message_json, "utf-8")
            )

            # Log resumido para o console
            cars_count = detection_result['vehicle_counts'].get('car', 0)
            total_count = detection_result['total_vehicles']
            
            logger.info(f"📤 Publicando contagem no tópico: {self.topic}")
            logger.info(f"🚗 {image_name}: {cars_count} carros, {total_count} veículos totais")

            operation = self.ipc_client.new_publish_to_iot_core()
            operation.activate(publish_request)
            operation.get_response().result(timeout=10.0)

            logger.info("✅ Contagem publicada com sucesso!")
            return True

        except Exception as e:
            logger.error(f"❌ Erro ao publicar contagem: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def run_car_counting_loop(self):
        """Loop principal de contagem de carros a cada 1 minuto"""
        logger.info("🚗 Iniciando loop de contagem de carros - processamento a cada 60 segundos...")
        
        cycle_count = 1
        
        try:
            while True:
                logger.info(f"🔄 Ciclo #{cycle_count} - Selecionando e processando imagem...")
                
                # Selecionar imagem aleatória
                selected_image = self.get_random_image()
                
                if selected_image:
                    # Detectar veículos
                    result = self.detect_vehicles(selected_image)
                    
                    if result:
                        # Salvar imagem anotada (opcional)
                        self.save_annotated_image(result['annotated_image'], selected_image.name)
                        
                        # Publicar resultado
                        success = self.publish_car_count(selected_image.name, result)
                        
                        if success:
                            cars = result['vehicle_counts'].get('car', 0)
                            total = result['total_vehicles']
                            logger.info(f"✅ Ciclo #{cycle_count}: {cars} carros detectados de {total} veículos totais")
                        else:
                            logger.error(f"❌ Falha ao publicar resultado do ciclo #{cycle_count}")
                    else:
                        logger.error(f"❌ Falha na detecção de veículos - ciclo #{cycle_count}")
                else:
                    logger.error(f"❌ Não foi possível selecionar imagem - ciclo #{cycle_count}")
                
                cycle_count += 1
                
                # Aguardar 1 minuto antes do próximo ciclo
                logger.info("⏳ Aguardando 60 segundos para próximo ciclo...")
                time.sleep(60)
                
        except KeyboardInterrupt:
            logger.info("🛑 Loop de contagem interrompido pelo usuário")
        except Exception as e:
            logger.error(f"❌ Erro no loop de contagem: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def run(self):
        """Loop principal do componente"""
        logger.info("🚀 Iniciando componente CarCounterGreengrass...")
        
        # Estabelecer conexão IPC
        if not self.connect_ipc():
            logger.error("❌ Falha ao conectar IPC. Encerrando...")
            return
        
        # Carregar modelo YOLO
        if not self.load_yolo_model():
            logger.error("❌ Falha ao carregar modelo YOLO. Encerrando...")
            return
        
        logger.info("🎯 Componente inicializado com sucesso!")
        logger.info("🚗 Modelo YOLO carregado - pronto para detectar veículos!")
        
        try:
            # Iniciar loop de contagem
            self.run_car_counting_loop()
                
        except KeyboardInterrupt:
            logger.info("🛑 Componente interrompido pelo usuário")
        except Exception as e:
            logger.error(f"❌ Erro no loop principal: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Limpeza de recursos"""
        logger.info("🧹 Realizando limpeza de recursos...")
        if self.ipc_client:
            try:
                logger.info("✅ Limpeza concluída")
            except Exception as e:
                logger.error(f"⚠️ Erro durante limpeza: {e}")

def main():
    """Função principal"""
    logger.info("=" * 60)
    logger.info("🚗 AWS IoT Greengrass V2 - Car Counter Component v1.0.6")
    logger.info("=" * 60)
    
    try:
        # Criar e executar componente
        component = GreengrassCarCounter()
        component.run()
        
    except Exception as e:
        logger.error(f"❌ Erro fatal no componente: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()
