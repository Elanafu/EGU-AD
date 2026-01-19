# EGU-AD

## Repository Structure

```text
.
├── main.py                     # Main entry for training and evaluation
├── data/
│   ├── __init__.py
│   └── data_loader.py          # Data loading and preprocessing
├── model/
│   ├── __init__.py
│   ├── temporal_encoder.py     # Temporal representation encoder
│   ├── eie_model.py            # Core anomaly detection model
│   ├── loss.py                 # Loss function definitions
│   └── anomaly_scoring.py      # Anomaly score computation
├── solver/
│   ├── __init__.py
│   └── solver_eie.py           # Training and evaluation pipeline
├── metrics/                    # Evaluation metrics (optional)
├── checkpoints/                # Model checkpoints
└── README.md


## Environment Requirements
Python ≥ 3.8
PyTorch ≥ 1.10
NumPy, SciPy, and other standard scientific Python libraries


## Training and Testing

### Training
python main.py     --mode train     --dataset MSL     --data_path ./dataset/MSL     --model_save_path cpt_eie     --lr 1e-4     --num_epochs 5     --batch_size 128     --win_size 110     --input_c 55     --d_model 512     --e_layers 2     --gpu 2    --anormly_ratio 2     --seed 2 

### Testing
python main.py     --mode test     --dataset MSL     --data_path ./dataset/MSL     --model_save_path cpt_eie     --lr 1e-4     --num_epochs 5     --batch_size 128     --win_size 110     --input_c 55     --d_model 512     --e_layers 2     --gpu 2    --anormly_ratio 2     --seed 2 
