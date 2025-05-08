ReID models:
```
# --- CLIP Models ---
python export_reid_weights.py --input-weights C:\Users\kritt\Downloads\spoton_ml\weights\reid\clip_duke.pt --output-weights C:\Users\kritt\Downloads\spoton_ml\weights\reid_resaved\clip_duke.pt --model-name clip --num-classes 702
python export_reid_weights.py --input-weights C:\Users\kritt\Downloads\spoton_ml\weights\reid\clip_market1501.pt --output-weights C:\Users\kritt\Downloads\spoton_ml\weights\reid_resaved\clip_market1501.pt --model-name clip --num-classes 751

# --- HACNN Models ---
python export_reid_weights.py --input-weights C:\Users\kritt\Downloads\spoton_ml\weights\reid\hacnn_dukemtmcreid.pt --output-weights C:\Users\kritt\Downloads\spoton_ml\weights\reid_resaved\hacnn_dukemtmcreid.pt --model-name hacnn --num-classes 702
python export_reid_weights.py --input-weights C:\Users\kritt\Downloads\spoton_ml\weights\reid\hacnn_market1501.pt --output-weights C:\Users\kritt\Downloads\spoton_ml\weights\reid_resaved\hacnn_market1501.pt --model-name hacnn --num-classes 751

# --- LMBN Models ---
python export_reid_weights.py --input-weights C:\Users\kritt\Downloads\spoton_ml\weights\reid\lmbn_n_duke.pt --output-weights C:\Users\kritt\Downloads\spoton_ml\weights\reid_resaved\lmbn_n_duke.pt --model-name lmbn_n --num-classes 702
python export_reid_weights.py --input-weights C:\Users\kritt\Downloads\spoton_ml\weights\reid\lmbn_n_market.pt --output-weights C:\Users\kritt\Downloads\spoton_ml\weights\reid_resaved\lmbn_n_market.pt --model-name lmbn_n --num-classes 751

# --- MLFN Models ---
python export_reid_weights.py --input-weights C:\Users\kritt\Downloads\spoton_ml\weights\reid\mlfn_dukemtmcreid.pt --output-weights C:\Users\kritt\Downloads\spoton_ml\weights\reid_resaved\mlfn_dukemtmcreid.pt --model-name mlfn --num-classes 702
python export_reid_weights.py --input-weights C:\Users\kritt\Downloads\spoton_ml\weights\reid\mlfn_market1501.pt --output-weights C:\Users\kritt\Downloads\spoton_ml\weights\reid_resaved\mlfn_market1501.pt --model-name mlfn --num-classes 751

# --- MobileNetV2 Models ---
python export_reid_weights.py --input-weights C:\Users\kritt\Downloads\spoton_ml\weights\reid\mobilenetv2_x1_4_dukemtmcreid.pt --output-weights C:\Users\kritt\Downloads\spoton_ml\weights\reid_resaved\mobilenetv2_x1_4_dukemtmcreid.pt --model-name mobilenetv2_x1_4 --num-classes 702
python export_reid_weights.py --input-weights C:\Users\kritt\Downloads\spoton_ml\weights\reid\mobilenetv2_x1_4_market1501.pt --output-weights C:\Users\kritt\Downloads\spoton_ml\weights\reid_resaved\mobilenetv2_x1_4_market1501.pt --model-name mobilenetv2_x1_4 --num-classes 751

# --- OSNet Models ---
python export_reid_weights.py --input-weights C:\Users\kritt\Downloads\spoton_ml\weights\reid\osnet_ain_x1_0_msmt17.pt --output-weights C:\Users\kritt\Downloads\spoton_ml\weights\reid_resaved\osnet_ain_x1_0_msmt17.pt --model-name osnet_ain_x1_0 --num-classes 1041 # Explicitly state MSMT17 classes
python export_reid_weights.py --input-weights C:\Users\kritt\Downloads\spoton_ml\weights\reid\osnet_ibn_x1_0_msmt17.pt --output-weights C:\Users\kritt\Downloads\spoton_ml\weights\reid_resaved\osnet_ibn_x1_0_msmt17.pt --model-name osnet_ibn_x1_0 --num-classes 1041 # Explicitly state MSMT17 classes
python export_reid_weights.py --input-weights C:\Users\kritt\Downloads\spoton_ml\weights\reid\osnet_x1_0_msmt17.pt --output-weights C:\Users\kritt\Downloads\spoton_ml\weights\reid_resaved\osnet_x1_0_msmt17.pt --model-name osnet_x1_0 --num-classes 1041 # Explicitly state MSMT17 classes

# --- ResNet50 Models ---
python export_reid_weights.py --input-weights C:\Users\kritt\Downloads\spoton_ml\weights\reid\resnet50_fc512_market1501.pt --output-weights C:\Users\kritt\Downloads\spoton_ml\weights\reid_resaved\resnet50_fc512_market1501.pt --model-name resnet50 --num-classes 751
python export_reid_weights.py --input-weights C:\Users\kritt\Downloads\spoton_ml\weights\reid\resnet50_fc512_msmt17.pt --output-weights C:\Users\kritt\Downloads\spoton_ml\weights\reid_resaved\resnet50_fc512_msmt17.pt --model-name resnet50 --num-classes 1041 # Explicitly state MSMT17 classes
```

Export-Import MLflow:
```
pip install git+https://github.com/mlflow/mlflow-export-import
```
then
```

```