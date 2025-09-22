#!/usr/bin/env python3
import json
import time
import logging
import sys
import traceback
import os
from datetime import datetime
from pathlib import Path

import torch
from torchvision import models, transforms
from PIL import Image

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

class GreengrassImageClassifier:
    def __init__(self):
        self.ipc_client = None
        self.component_name = "com.example.ImageClassifierGreengrass"
        self.topic = "greengrass/v2/image-classifier/results"
        
        # Configurações do classificador
        self.base_dir = "/greengrass/v2/packages/artifacts/com.example.ImageClassifierGreengrass/1.0.5/data/images"
        self.model = None
        self.labels = []
        self.transform = None
        
        # ✅ CORREÇÃO: Configurar cache do PyTorch em diretório com permissão
        self.cache_dir = "/tmp/pytorch_cache"
        self.setup_cache_directory()
        
    def setup_cache_directory(self):
        """Configura diretório de cache do PyTorch com permissões corretas"""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Configurar variáveis de ambiente para PyTorch usar nosso cache
            os.environ['TORCH_HOME'] = self.cache_dir
            os.environ['TORCH_CACHE'] = self.cache_dir
            
            logger.info(f"✅ Diretório de cache configurado: {self.cache_dir}")
        except Exception as e:
            logger.error(f"❌ Erro ao configurar cache: {e}")
            # Tentar usar /tmp como fallback
            self.cache_dir = "/tmp"
            os.environ['TORCH_HOME'] = self.cache_dir
            logger.info(f"⚠️ Usando /tmp como diretório de cache: {self.cache_dir}")
        
    def connect_ipc(self):
        """Estabelece conexão IPC com o Greengrass Core"""
        try:
            logger.info("🔄 Tentando estabelecer conexão IPC...")
            
            # Método correto para estabelecer conexão IPC no Greengrass V2
            self.ipc_client = awsiot.greengrasscoreipc.connect()
            
            logger.info("✅ Conexão IPC estabelecida com sucesso!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao estabelecer conexão IPC: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def load_model(self):
        """Carrega o modelo ResNet18 com configuração de cache personalizada"""
        try:
            logger.info("🔄 Carregando modelo ResNet18...")
            logger.info(f"📁 Cache directory: {os.environ.get('TORCH_HOME', 'default')}")
            
            # ✅ CORREÇÃO: Usar weights parameter em vez de pretrained (deprecated)
            from torchvision.models import ResNet18_Weights
            self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.model.eval()
            
            logger.info("✅ Modelo ResNet18 carregado com sucesso!")
            return True
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # ✅ FALLBACK: Tentar carregar sem pesos pré-treinados se houver problema
            try:
                logger.info("🔄 Tentando carregar modelo sem pesos pré-treinados...")
                self.model = models.resnet18(weights=None)
                self.model.eval()
                logger.warning("⚠️ Modelo carregado SEM pesos pré-treinados - classificações serão aleatórias")
                return True
            except Exception as e2:
                logger.error(f"❌ Erro também no fallback: {e2}")
                return False
    
    def setup_transforms(self):
        """Configura as transformações para as imagens"""
        try:
            logger.info("🔄 Configurando transformações de imagem...")
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            logger.info("✅ Transformações configuradas com sucesso!")
            return True
        except Exception as e:
            logger.error(f"❌ Erro ao configurar transformações: {e}")
            return False
    
    def load_labels(self):
        """Carrega os labels do ImageNet"""
        try:
            logger.info("🔄 Carregando labels do ImageNet...")
            
            # ✅ CORREÇÃO: Usar diretório de cache configurado
            labels_path = os.path.join(self.cache_dir, "imagenet_classes.txt")
            
            url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
            
            if not os.path.exists(labels_path):
                logger.info("📥 Baixando arquivo de labels...")
                import urllib.request
                try:
                    urllib.request.urlretrieve(url, labels_path)
                except Exception as e:
                    logger.error(f"❌ Erro ao baixar labels: {e}")
                    # Fallback: usar labels básicos
                    logger.info("📝 Usando labels básicos como fallback...")
                    self.labels = [f"class_{i}" for i in range(1000)]
                    return True
            
            with open(labels_path) as f:
                self.labels = [line.strip() for line in f.readlines()]
            
            logger.info(f"✅ {len(self.labels)} labels carregados com sucesso!")
            return True
        except Exception as e:
            logger.error(f"❌ Erro ao carregar labels: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Fallback básico
            self.labels = [f"class_{i}" for i in range(1000)]
            logger.info("📝 Usando labels genéricos como fallback")
            return True
    
    def classify_image(self, image_path):
        """Classifica uma imagem usando o modelo"""
        try:
            # Carregar e processar imagem
            img = Image.open(image_path).convert("RGB")
            input_tensor = self.transform(img).unsqueeze(0)  # adiciona batch dimension
            
            # Inferência
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                _, predicted = torch.max(outputs, 1)
                confidence = probabilities[predicted.item()].item()
            
            predicted_label = self.labels[predicted.item()]
            
            return {
                'label': predicted_label,
                'confidence': float(confidence),
                'class_id': int(predicted.item())
            }
            
        except Exception as e:
            logger.error(f"❌ Erro ao classificar imagem {image_path}: {e}")
            return None
    
    def publish_result(self, image_name, classification_result):
        """Publica resultado da classificação via IPC"""
        try:
            if not self.ipc_client:
                logger.error("❌ Cliente IPC não está conectado")
                return False

            message_data = {
                "timestamp": datetime.now().isoformat(),
                "component": self.component_name,
                "image_name": image_name,
                "classification": classification_result,
                "thing_name": "MeuCoreWSLDockerV2"
            }

            message_json = json.dumps(message_data)

            publish_request = PublishToIoTCoreRequest(
                topic_name=self.topic,
                qos=QOS.AT_LEAST_ONCE,
                payload=bytes(message_json, "utf-8")
            )

            logger.info(f"📤 Publicando resultado no tópico: {self.topic}")
            logger.info(f"📄 Imagem: {image_name} -> {classification_result['label']} ({classification_result['confidence']:.2%})")

            operation = self.ipc_client.new_publish_to_iot_core()
            operation.activate(publish_request)
            operation.get_response().result(timeout=10.0)

            logger.info("✅ Resultado publicado com sucesso!")
            return True

        except Exception as e:
            logger.error(f"❌ Erro ao publicar resultado: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def process_images(self):
        """Processa todas as imagens do diretório"""
        images_dir = Path(self.base_dir)
        
        if not images_dir.exists():
            logger.error(f"❌ Diretório {self.base_dir} não existe")
            return False
        
        # Buscar imagens (jpg, jpeg, png)
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_files = []
        
        for extension in image_extensions:
            image_files.extend(images_dir.glob(extension))
        
        if not image_files:
            logger.warning(f"⚠️ Nenhuma imagem encontrada em {self.base_dir}")
            return False
        
        logger.info(f"📁 Encontradas {len(image_files)} imagens para processar")
        
        processed_count = 0
        error_count = 0
        
        for image_file in image_files:
            try:
                logger.info(f"🔄 Processando: {image_file.name}")
                
                # Classificar imagem
                result = self.classify_image(image_file)
                
                if result:
                    # Publicar resultado
                    success = self.publish_result(image_file.name, result)
                    
                    if success:
                        processed_count += 1
                        logger.info(f"✅ {image_file.name} processada com sucesso")
                    else:
                        error_count += 1
                        logger.error(f"❌ Falha ao publicar resultado para {image_file.name}")
                else:
                    error_count += 1
                    logger.error(f"❌ Falha na classificação de {image_file.name}")
                
                # Aguardar um pouco entre processamentos
                time.sleep(2)
                
            except Exception as e:
                error_count += 1
                logger.error(f"❌ Erro ao processar {image_file.name}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
        
        logger.info(f"📊 Processamento concluído: {processed_count} sucessos, {error_count} erros")
        return processed_count > 0
    
    def run_continuous(self):
        """Executa processamento contínuo (monitora diretório)"""
        logger.info("🔄 Iniciando modo contínuo - monitorando diretório...")
        
        processed_files = set()
        
        try:
            while True:
                images_dir = Path(self.base_dir)
                
                if images_dir.exists():
                    # Buscar novas imagens
                    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
                    current_files = set()
                    
                    for extension in image_extensions:
                        current_files.update(images_dir.glob(extension))
                    
                    # Processar apenas arquivos novos
                    new_files = current_files - processed_files
                    
                    if new_files:
                        logger.info(f"📁 Encontradas {len(new_files)} novas imagens")
                        
                        for image_file in new_files:
                            try:
                                logger.info(f"🔄 Processando nova imagem: {image_file.name}")
                                
                                # Classificar e publicar
                                result = self.classify_image(image_file)
                                
                                if result:
                                    self.publish_result(image_file.name, result)
                                    processed_files.add(image_file)
                                    logger.info(f"✅ {image_file.name} processada e adicionada ao conjunto")
                                
                                time.sleep(1)
                                
                            except Exception as e:
                                logger.error(f"❌ Erro ao processar nova imagem {image_file.name}: {e}")
                
                # Aguardar antes da próxima verificação
                time.sleep(30)
                
        except KeyboardInterrupt:
            logger.info("🛑 Processamento contínuo interrompido pelo usuário")
        except Exception as e:
            logger.error(f"❌ Erro no processamento contínuo: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def run(self):
        """Loop principal do componente"""
        logger.info("🚀 Iniciando componente ImageClassifierGreengrass...")
        
        # Estabelecer conexão IPC
        if not self.connect_ipc():
            logger.error("❌ Falha ao conectar IPC. Encerrando...")
            return
        
        # Carregar modelo
        if not self.load_model():
            logger.error("❌ Falha ao carregar modelo. Encerrando...")
            return
        
        # Configurar transformações
        if not self.setup_transforms():
            logger.error("❌ Falha ao configurar transformações. Encerrando...")
            return
        
        # Carregar labels
        if not self.load_labels():
            logger.error("❌ Falha ao carregar labels. Encerrando...")
            return
        
        logger.info("🎯 Componente inicializado com sucesso!")
        
        try:
            # Processar imagens existentes uma vez
            logger.info("🔄 Processando imagens existentes...")
            self.process_images()
            
            # Depois entrar em modo contínuo
            self.run_continuous()
                
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
                # O cliente IPC é gerenciado automaticamente pelo SDK
                logger.info("✅ Limpeza concluída")
            except Exception as e:
                logger.error(f"⚠️ Erro durante limpeza: {e}")

def main():
    """Função principal"""
    logger.info("=" * 60)
    logger.info("🖼️ AWS IoT Greengrass V2 - Image Classifier Component v1.0.5")
    logger.info("=" * 60)
    
    try:
        # Criar e executar componente
        component = GreengrassImageClassifier()
        component.run()
        
    except Exception as e:
        logger.error(f"❌ Erro fatal no componente: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()
