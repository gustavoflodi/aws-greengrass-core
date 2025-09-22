#!/usr/bin/env python3
import json
import time
import logging
import sys
import traceback
from datetime import datetime

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

class GreengrassHelloWorld:
    def __init__(self):
        self.ipc_client = None
        self.component_name = "com.example.HelloWorldGreengrass"
        self.topic = "greengrass/v2/hello"
        
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
    
    def publish_message(self, message_data):
        try:
            if not self.ipc_client:
                logger.error("❌ Cliente IPC não está conectado")
                return False

            message_json = json.dumps(message_data)

            publish_request = PublishToIoTCoreRequest(
                topic_name=self.topic,
                qos=QOS.AT_LEAST_ONCE,
                payload=bytes(message_json, "utf-8")  # ✅ correto para IoT Core
            )

            logger.info(f"📤 Publicando mensagem no tópico: {self.topic}")
            logger.info(f"📄 Conteúdo: {message_json}")

            operation = self.ipc_client.new_publish_to_iot_core()
            operation.activate(publish_request)
            operation.get_response().result(timeout=10.0)

            logger.info("✅ Mensagem publicada com sucesso!")
            return True

        except Exception as e:
            logger.error(f"❌ Erro ao publicar mensagem: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

        except Exception as e:
            logger.error(f"❌ Erro ao publicar mensagem: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    
    def run(self):
        """Loop principal do componente"""
        logger.info("🚀 Iniciando componente HelloWorldGreengrass...")
        
        # Estabelecer conexão IPC
        if not self.connect_ipc():
            logger.error("❌ Falha ao conectar IPC. Encerrando...")
            return
        
        # Contador de mensagens
        message_count = 1
        
        try:
            while True:
                # Preparar dados da mensagem
                message_data = {
                    "timestamp": datetime.now().isoformat(),
                    "component": self.component_name,
                    "message": f"Hello from Greengrass V2! Message #{message_count}",
                    "thing_name": "MeuCoreWSLDockerV2",
                    "count": message_count
                }
                
                # Publicar mensagem
                success = self.publish_message(message_data)
                
                if success:
                    logger.info(f"✅ Mensagem #{message_count} enviada com sucesso")
                else:
                    logger.error(f"❌ Falha ao enviar mensagem #{message_count}")
                
                message_count += 1
                
                # Aguardar 30 segundos antes da próxima mensagem
                logger.info("⏳ Aguardando 30 segundos...")
                time.sleep(30)
                
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
    logger.info("=" * 50)
    logger.info("🌱 AWS IoT Greengrass V2 - Hello World Component")
    logger.info("=" * 50)
    
    try:
        # Criar e executar componente
        component = GreengrassHelloWorld()
        component.run()
        
    except Exception as e:
        logger.error(f"❌ Erro fatal no componente: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()
