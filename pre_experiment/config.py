from dataclasses import dataclass

@dataclass
class BaseConfig:
    # 通用硬件与训练参数
    model_name_or_path: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    batch_size: int = 16
    lr: float = 3e-5
    epochs: int = 20
    device: str = "cuda"
    max_len: int = 128
    
@dataclass
class NERConfig(BaseConfig):
    # NER 任务专属路径与配置
    task_name: str = "NER"
    train_data_path: str = "../data/data_pre/data_pred/BC5CDR/merged_bc5cdr_train_data.json"
    val_data_path: str = "../data/data_pre/data_pred/BC5CDR/merged_bc5cdr_development_data.json"
    test_data_path: str = "../data/data_pre/data_pred/BC5CDR/merged_bc5cdr_test_data.json"
    num_labels: int = 5  # O, B-Chemical, I-Chemical, B-Disease, I-Disease
    
    # 对比学习参数
    proj_dim: int = 128
    alpha: float = 0.5  # CE loss 和 CL loss 的权重
    temperature: float = 0.07
