"""
Deployment Configuration and Container Setup for RF-DETR Surveillance
Docker, Kubernetes, and cloud deployment configurations
"""
import logging
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
import tempfile
import shutil
import subprocess

logger = logging.getLogger(__name__)


@dataclass
class ResourceLimits:
    """Resource limits configuration."""
    
    cpu_request: str = "500m"  # CPU request (millicores)
    cpu_limit: str = "2000m"   # CPU limit (millicores)
    memory_request: str = "1Gi"  # Memory request
    memory_limit: str = "4Gi"    # Memory limit
    gpu_limit: int = 0  # GPU count (0 for CPU-only)


@dataclass
class ContainerConfig:
    """Container configuration for deployment."""
    
    # Base image settings
    base_image: str = "python:3.9-slim"
    cuda_base_image: str = "nvidia/cuda:11.8-runtime-ubuntu20.04"
    use_gpu: bool = False
    
    # Application settings
    app_name: str = "rfdetr-surveillance"
    app_version: str = "1.0.0"
    working_dir: str = "/app"
    
    # Dependencies
    python_version: str = "3.9"
    torch_version: str = "2.0.0"
    torchvision_version: str = "0.15.0"
    additional_packages: List[str] = field(default_factory=lambda: [
        "fastapi>=0.104.0",
        "uvicorn>=0.23.0",
        "pillow>=9.0.0",
        "numpy>=1.21.0",
        "psutil>=5.8.0",
        "pydantic>=2.0.0"
    ])
    
    # Ports
    http_port: int = 8000
    metrics_port: int = 9090
    
    # Resource limits
    resources: ResourceLimits = field(default_factory=ResourceLimits)
    
    # Health check
    health_check_path: str = "/health"
    health_check_interval: int = 30
    health_check_timeout: int = 5
    
    # Environment variables
    environment_vars: Dict[str, str] = field(default_factory=dict)


@dataclass
class KubernetesConfig:
    """Kubernetes deployment configuration."""
    
    # Deployment settings
    namespace: str = "default"
    replica_count: int = 3
    rolling_update_strategy: Dict[str, str] = field(default_factory=lambda: {
        "type": "RollingUpdate",
        "maxSurge": "25%",
        "maxUnavailable": "25%"
    })
    
    # Service settings
    service_type: str = "LoadBalancer"  # ClusterIP, NodePort, LoadBalancer
    service_port: int = 80
    target_port: int = 8000
    
    # Ingress settings
    enable_ingress: bool = False
    ingress_host: str = "rfdetr-api.example.com"
    ingress_class: str = "nginx"
    enable_tls: bool = False
    tls_secret_name: str = "rfdetr-tls"
    
    # Storage
    enable_persistent_storage: bool = False
    storage_class: str = "standard"
    storage_size: str = "10Gi"
    mount_path: str = "/data"
    
    # Auto-scaling
    enable_hpa: bool = True  # Horizontal Pod Autoscaler
    min_replicas: int = 2
    max_replicas: int = 10
    target_cpu_percentage: int = 70
    target_memory_percentage: int = 80
    
    # Monitoring
    enable_service_monitor: bool = False  # For Prometheus
    metrics_path: str = "/metrics"
    
    # Node selection
    node_selector: Dict[str, str] = field(default_factory=dict)
    tolerations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Security
    security_context: Dict[str, Any] = field(default_factory=lambda: {
        "runAsNonRoot": True,
        "runAsUser": 1000,
        "readOnlyRootFilesystem": False
    })


@dataclass
class CloudConfig:
    """Cloud deployment configuration."""
    
    # Cloud provider
    provider: str = "aws"  # aws, gcp, azure
    region: str = "us-west-2"
    
    # Container registry
    registry_url: str = ""  # e.g., "123456789012.dkr.ecr.us-west-2.amazonaws.com"
    image_repository: str = "rfdetr-surveillance"
    
    # Load balancing
    load_balancer_type: str = "application"  # application, network
    enable_sticky_sessions: bool = False
    
    # Auto-scaling
    enable_cluster_autoscaler: bool = True
    instance_types: List[str] = field(default_factory=lambda: ["m5.large", "m5.xlarge"])
    
    # Monitoring and logging
    enable_cloudwatch: bool = True
    enable_prometheus: bool = False
    log_retention_days: int = 7


class DockerfileGenerator:
    """Generate optimized Dockerfiles for RF-DETR deployment."""
    
    def __init__(self, config: ContainerConfig):
        self.config = config
    
    def generate_dockerfile(self, output_path: Optional[Path] = None) -> str:
        """Generate Dockerfile content."""
        
        base_image = self.config.cuda_base_image if self.config.use_gpu else self.config.base_image
        
        dockerfile_content = f"""# RF-DETR Surveillance API Dockerfile
# Generated automatically for deployment

FROM {base_image}

# Set working directory
WORKDIR {self.config.working_dir}

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    wget \\
    curl \\
    git \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Install Python if using CUDA base
"""
        
        if self.config.use_gpu and "cuda" in self.config.cuda_base_image:
            dockerfile_content += f"""
# Install Python {self.config.python_version}
RUN apt-get update && apt-get install -y \\
    python{self.config.python_version} \\
    python{self.config.python_version}-pip \\
    python{self.config.python_version}-dev \\
    && ln -s /usr/bin/python{self.config.python_version} /usr/bin/python \\
    && ln -s /usr/bin/pip{self.config.python_version} /usr/bin/pip
"""
        
        # PyTorch installation
        torch_install_cmd = self._get_torch_install_command()
        
        dockerfile_content += f"""
# Install PyTorch
{torch_install_cmd}

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY configs/ ./configs/

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser {self.config.working_dir}

# Set environment variables
ENV PYTHONPATH={self.config.working_dir}
ENV MODEL_PATH=./models/model.pt
ENV DEVICE=auto
ENV LOG_LEVEL=INFO
"""

        # Add custom environment variables
        for key, value in self.config.environment_vars.items():
            dockerfile_content += f"ENV {key}={value}\n"

        dockerfile_content += f"""
# Expose ports
EXPOSE {self.config.http_port}
EXPOSE {self.config.metrics_port}

# Health check
HEALTHCHECK --interval={self.config.health_check_interval}s \\
    --timeout={self.config.health_check_timeout}s \\
    --start-period=30s \\
    --retries=3 \\
    CMD curl -f http://localhost:{self.config.http_port}{self.config.health_check_path} || exit 1

# Switch to non-root user
USER appuser

# Run the application
CMD ["python", "-m", "uvicorn", "src.components.deployment.inference_server:app", "--host", "0.0.0.0", "--port", "{self.config.http_port}"]
"""
        
        if output_path:
            output_path.write_text(dockerfile_content)
            logger.info(f"Dockerfile generated: {output_path}")
        
        return dockerfile_content
    
    def _get_torch_install_command(self) -> str:
        """Get PyTorch installation command based on configuration."""
        
        if self.config.use_gpu:
            # CUDA version
            return f"""RUN pip install --no-cache-dir torch=={self.config.torch_version} \\
    torchvision=={self.config.torchvision_version} \\
    --index-url https://download.pytorch.org/whl/cu118"""
        else:
            # CPU version
            return f"""RUN pip install --no-cache-dir torch=={self.config.torch_version} \\
    torchvision=={self.config.torchvision_version} \\
    --index-url https://download.pytorch.org/whl/cpu"""
    
    def generate_requirements_txt(self, output_path: Optional[Path] = None) -> str:
        """Generate requirements.txt file."""
        
        requirements = self.config.additional_packages.copy()
        
        # Add core dependencies
        core_deps = [
            "torch>=2.0.0",
            "torchvision>=0.15.0", 
            "onnx>=1.14.0",
            "onnxruntime>=1.15.0",
            "scipy>=1.9.0"
        ]
        
        requirements.extend(core_deps)
        
        # Remove duplicates and sort
        unique_requirements = sorted(list(set(requirements)))
        
        content = "\n".join(unique_requirements)
        
        if output_path:
            output_path.write_text(content)
            logger.info(f"Requirements.txt generated: {output_path}")
        
        return content
    
    def generate_dockerignore(self, output_path: Optional[Path] = None) -> str:
        """Generate .dockerignore file."""
        
        dockerignore_content = """# RF-DETR Deployment .dockerignore

# Git
.git/
.gitignore
.gitattributes

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
.env

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Training artifacts (keep only final models)
training_logs/
checkpoints/
runs/
wandb/

# Documentation
docs/
*.md
LICENSE

# Tests
tests/
pytest.ini
.pytest_cache/

# Development files
Makefile
docker-compose*.yml
kubernetes/
deployment/

# Large data files
data/
datasets/
*.zip
*.tar.gz
*.h5
*.hdf5
"""
        
        if output_path:
            output_path.write_text(dockerignore_content)
            logger.info(f".dockerignore generated: {output_path}")
        
        return dockerignore_content


class KubernetesManifestGenerator:
    """Generate Kubernetes deployment manifests."""
    
    def __init__(self, container_config: ContainerConfig, k8s_config: KubernetesConfig):
        self.container_config = container_config
        self.k8s_config = k8s_config
    
    def generate_deployment(self) -> Dict[str, Any]:
        """Generate Kubernetes Deployment manifest."""
        
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": self.container_config.app_name,
                "namespace": self.k8s_config.namespace,
                "labels": {
                    "app": self.container_config.app_name,
                    "version": self.container_config.app_version
                }
            },
            "spec": {
                "replicas": self.k8s_config.replica_count,
                "strategy": self.k8s_config.rolling_update_strategy,
                "selector": {
                    "matchLabels": {
                        "app": self.container_config.app_name
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": self.container_config.app_name,
                            "version": self.container_config.app_version
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": self.container_config.app_name,
                            "image": f"{self.container_config.app_name}:{self.container_config.app_version}",
                            "ports": [
                                {
                                    "containerPort": self.container_config.http_port,
                                    "name": "http"
                                },
                                {
                                    "containerPort": self.container_config.metrics_port,
                                    "name": "metrics"
                                }
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": self.container_config.resources.cpu_request,
                                    "memory": self.container_config.resources.memory_request
                                },
                                "limits": {
                                    "cpu": self.container_config.resources.cpu_limit,
                                    "memory": self.container_config.resources.memory_limit
                                }
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": self.container_config.health_check_path,
                                    "port": self.container_config.http_port
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 30
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": self.container_config.health_check_path,
                                    "port": self.container_config.http_port
                                },
                                "initialDelaySeconds": 10,
                                "periodSeconds": 5
                            }
                        }],
                        "securityContext": self.k8s_config.security_context
                    }
                }
            }
        }
        
        # Add GPU resources if needed
        if self.container_config.resources.gpu_limit > 0:
            deployment["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] = self.container_config.resources.gpu_limit
        
        # Add node selector
        if self.k8s_config.node_selector:
            deployment["spec"]["template"]["spec"]["nodeSelector"] = self.k8s_config.node_selector
        
        # Add tolerations
        if self.k8s_config.tolerations:
            deployment["spec"]["template"]["spec"]["tolerations"] = self.k8s_config.tolerations
        
        return deployment
    
    def generate_service(self) -> Dict[str, Any]:
        """Generate Kubernetes Service manifest."""
        
        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{self.container_config.app_name}-service",
                "namespace": self.k8s_config.namespace,
                "labels": {
                    "app": self.container_config.app_name
                }
            },
            "spec": {
                "type": self.k8s_config.service_type,
                "ports": [
                    {
                        "port": self.k8s_config.service_port,
                        "targetPort": self.k8s_config.target_port,
                        "protocol": "TCP",
                        "name": "http"
                    }
                ],
                "selector": {
                    "app": self.container_config.app_name
                }
            }
        }
        
        return service
    
    def generate_hpa(self) -> Optional[Dict[str, Any]]:
        """Generate Horizontal Pod Autoscaler manifest."""
        
        if not self.k8s_config.enable_hpa:
            return None
        
        hpa = {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"{self.container_config.app_name}-hpa",
                "namespace": self.k8s_config.namespace
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": self.container_config.app_name
                },
                "minReplicas": self.k8s_config.min_replicas,
                "maxReplicas": self.k8s_config.max_replicas,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": self.k8s_config.target_cpu_percentage
                            }
                        }
                    },
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "memory",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": self.k8s_config.target_memory_percentage
                            }
                        }
                    }
                ]
            }
        }
        
        return hpa
    
    def generate_ingress(self) -> Optional[Dict[str, Any]]:
        """Generate Ingress manifest."""
        
        if not self.k8s_config.enable_ingress:
            return None
        
        ingress = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {
                "name": f"{self.container_config.app_name}-ingress",
                "namespace": self.k8s_config.namespace,
                "annotations": {
                    "kubernetes.io/ingress.class": self.k8s_config.ingress_class
                }
            },
            "spec": {
                "rules": [
                    {
                        "host": self.k8s_config.ingress_host,
                        "http": {
                            "paths": [
                                {
                                    "path": "/",
                                    "pathType": "Prefix",
                                    "backend": {
                                        "service": {
                                            "name": f"{self.container_config.app_name}-service",
                                            "port": {
                                                "number": self.k8s_config.service_port
                                            }
                                        }
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        }
        
        # Add TLS if enabled
        if self.k8s_config.enable_tls:
            ingress["spec"]["tls"] = [
                {
                    "hosts": [self.k8s_config.ingress_host],
                    "secretName": self.k8s_config.tls_secret_name
                }
            ]
        
        return ingress


class DeploymentManager:
    """Comprehensive deployment manager for RF-DETR surveillance."""
    
    def __init__(
        self, 
        container_config: ContainerConfig,
        k8s_config: Optional[KubernetesConfig] = None,
        cloud_config: Optional[CloudConfig] = None
    ):
        self.container_config = container_config
        self.k8s_config = k8s_config or KubernetesConfig()
        self.cloud_config = cloud_config or CloudConfig()
        
        self.dockerfile_generator = DockerfileGenerator(container_config)
        if k8s_config:
            self.k8s_generator = KubernetesManifestGenerator(container_config, k8s_config)
    
    def generate_deployment_files(self, output_dir: Path) -> Dict[str, Path]:
        """Generate all deployment files."""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generated_files = {}
        
        # Docker files
        docker_dir = output_dir / "docker"
        docker_dir.mkdir(exist_ok=True)
        
        dockerfile_path = docker_dir / "Dockerfile"
        self.dockerfile_generator.generate_dockerfile(dockerfile_path)
        generated_files["dockerfile"] = dockerfile_path
        
        requirements_path = docker_dir / "requirements.txt"
        self.dockerfile_generator.generate_requirements_txt(requirements_path)
        generated_files["requirements"] = requirements_path
        
        dockerignore_path = docker_dir / ".dockerignore"
        self.dockerfile_generator.generate_dockerignore(dockerignore_path)
        generated_files["dockerignore"] = dockerignore_path
        
        # Kubernetes manifests
        if self.k8s_config:
            k8s_dir = output_dir / "kubernetes"
            k8s_dir.mkdir(exist_ok=True)
            
            # Deployment
            deployment = self.k8s_generator.generate_deployment()
            deployment_path = k8s_dir / "deployment.yaml"
            with open(deployment_path, 'w') as f:
                yaml.dump(deployment, f, default_flow_style=False)
            generated_files["k8s_deployment"] = deployment_path
            
            # Service
            service = self.k8s_generator.generate_service()
            service_path = k8s_dir / "service.yaml"
            with open(service_path, 'w') as f:
                yaml.dump(service, f, default_flow_style=False)
            generated_files["k8s_service"] = service_path
            
            # HPA
            hpa = self.k8s_generator.generate_hpa()
            if hpa:
                hpa_path = k8s_dir / "hpa.yaml"
                with open(hpa_path, 'w') as f:
                    yaml.dump(hpa, f, default_flow_style=False)
                generated_files["k8s_hpa"] = hpa_path
            
            # Ingress
            ingress = self.k8s_generator.generate_ingress()
            if ingress:
                ingress_path = k8s_dir / "ingress.yaml"
                with open(ingress_path, 'w') as f:
                    yaml.dump(ingress, f, default_flow_style=False)
                generated_files["k8s_ingress"] = ingress_path
        
        # Generate deployment scripts
        scripts_dir = output_dir / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        # Docker build script
        build_script = self._generate_build_script()
        build_script_path = scripts_dir / "build.sh"
        build_script_path.write_text(build_script)
        build_script_path.chmod(0o755)
        generated_files["build_script"] = build_script_path
        
        # Kubernetes deployment script
        if self.k8s_config:
            deploy_script = self._generate_deploy_script()
            deploy_script_path = scripts_dir / "deploy.sh"
            deploy_script_path.write_text(deploy_script)
            deploy_script_path.chmod(0o755)
            generated_files["deploy_script"] = deploy_script_path
        
        logger.info(f"Generated {len(generated_files)} deployment files in {output_dir}")
        
        return generated_files
    
    def _generate_build_script(self) -> str:
        """Generate Docker build script."""
        
        script = f"""#!/bin/bash
# RF-DETR Surveillance Docker Build Script

set -e

APP_NAME="{self.container_config.app_name}"
VERSION="{self.container_config.app_version}"
DOCKERFILE="docker/Dockerfile"

echo "Building RF-DETR Surveillance Docker image..."
echo "App: $APP_NAME"
echo "Version: $VERSION"

# Build the image
docker build -f $DOCKERFILE -t $APP_NAME:$VERSION .
docker tag $APP_NAME:$VERSION $APP_NAME:latest

echo "Build completed successfully!"
echo "Image: $APP_NAME:$VERSION"

# Optional: Push to registry
if [ "$1" == "--push" ]; then
    if [ -n "$REGISTRY_URL" ]; then
        echo "Pushing to registry: $REGISTRY_URL"
        docker tag $APP_NAME:$VERSION $REGISTRY_URL/$APP_NAME:$VERSION
        docker push $REGISTRY_URL/$APP_NAME:$VERSION
        docker tag $APP_NAME:$VERSION $REGISTRY_URL/$APP_NAME:latest
        docker push $REGISTRY_URL/$APP_NAME:latest
        echo "Push completed!"
    else
        echo "REGISTRY_URL not set. Skipping push."
    fi
fi
"""
        
        return script
    
    def _generate_deploy_script(self) -> str:
        """Generate Kubernetes deployment script."""
        
        script = f"""#!/bin/bash
# RF-DETR Surveillance Kubernetes Deployment Script

set -e

NAMESPACE="{self.k8s_config.namespace}"
APP_NAME="{self.container_config.app_name}"

echo "Deploying RF-DETR Surveillance to Kubernetes..."
echo "Namespace: $NAMESPACE"
echo "App: $APP_NAME"

# Create namespace if it doesn't exist
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Apply Kubernetes manifests
echo "Applying Kubernetes manifests..."

kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml

# Apply HPA if exists
if [ -f "kubernetes/hpa.yaml" ]; then
    kubectl apply -f kubernetes/hpa.yaml
fi

# Apply Ingress if exists
if [ -f "kubernetes/ingress.yaml" ]; then
    kubectl apply -f kubernetes/ingress.yaml
fi

# Wait for deployment to be ready
echo "Waiting for deployment to be ready..."
kubectl rollout status deployment/$APP_NAME -n $NAMESPACE

# Show deployment status
echo "Deployment completed!"
kubectl get pods -n $NAMESPACE -l app=$APP_NAME
kubectl get services -n $NAMESPACE -l app=$APP_NAME

echo "RF-DETR Surveillance deployed successfully!"
"""
        
        return script
    
    def get_deployment_summary(self) -> Dict[str, Any]:
        """Get deployment configuration summary."""
        
        summary = {
            "container_config": {
                "app_name": self.container_config.app_name,
                "version": self.container_config.app_version,
                "base_image": self.container_config.base_image,
                "use_gpu": self.container_config.use_gpu,
                "ports": {
                    "http": self.container_config.http_port,
                    "metrics": self.container_config.metrics_port
                },
                "resources": asdict(self.container_config.resources)
            },
            "kubernetes_config": {
                "namespace": self.k8s_config.namespace,
                "replicas": self.k8s_config.replica_count,
                "service_type": self.k8s_config.service_type,
                "auto_scaling": self.k8s_config.enable_hpa,
                "ingress_enabled": self.k8s_config.enable_ingress
            },
            "cloud_config": {
                "provider": self.cloud_config.provider,
                "region": self.cloud_config.region,
                "registry": self.cloud_config.registry_url
            }
        }
        
        return summary


def create_deployment_manager(
    app_name: str = "rfdetr-surveillance",
    app_version: str = "1.0.0",
    use_gpu: bool = False,
    enable_kubernetes: bool = True,
    **kwargs
) -> DeploymentManager:
    """
    Create deployment manager with configuration.
    
    Args:
        app_name: Application name
        app_version: Application version
        use_gpu: Whether to use GPU for inference
        enable_kubernetes: Enable Kubernetes configuration
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured deployment manager
    """
    
    container_config = ContainerConfig(
        app_name=app_name,
        app_version=app_version,
        use_gpu=use_gpu,
        **kwargs
    )
    
    k8s_config = KubernetesConfig() if enable_kubernetes else None
    cloud_config = CloudConfig()
    
    return DeploymentManager(container_config, k8s_config, cloud_config)


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing Deployment Configuration")
    
    try:
        # Create deployment manager
        manager = create_deployment_manager(
            app_name="rfdetr-surveillance",
            app_version="1.0.0",
            use_gpu=True,
            enable_kubernetes=True
        )
        
        print("✅ Deployment manager created")
        
        # Generate deployment files
        output_dir = Path("test_deployment")
        generated_files = manager.generate_deployment_files(output_dir)
        
        print(f"✅ Generated {len(generated_files)} deployment files")
        
        for file_type, file_path in generated_files.items():
            print(f"  📄 {file_type}: {file_path}")
        
        # Get deployment summary
        summary = manager.get_deployment_summary()
        print("📊 Deployment Summary:")
        print(f"  App: {summary['container_config']['app_name']} v{summary['container_config']['version']}")
        print(f"  GPU: {summary['container_config']['use_gpu']}")
        print(f"  Replicas: {summary['kubernetes_config']['replicas']}")
        print(f"  Auto-scaling: {summary['kubernetes_config']['auto_scaling']}")
        
        # Cleanup
        if output_dir.exists():
            shutil.rmtree(output_dir)
        
        print("✅ Deployment configuration testing completed")
        
    except Exception as e:
        print(f"❌ Deployment configuration test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("✅ Deployment Configuration testing completed")