set -e

ROOT_DIR="$(pwd)"
ML_DIR="$ROOT_DIR/ml"
SCRATCH_DIR="$ROOT_DIR/scratch"
FILES_STORAGE="$ROOT_DIR/files"

rm -f "$FILES_STORAGE/channel_metrics.log" "$FILES_STORAGE/throughput.log"

echo ">>> [0/6] Передача тестового файла для оценки качества канала"
echo "TEST-DATA" > "$FILES_STORAGE/dummy_test.bin"
./ns3 run "scratch/wifi-sim --inputFile=$FILES_STORAGE/dummy_test.bin --outputFile=$FILES_STORAGE/dummy_result.bin"

echo ">>> [1/6] Выбор split-точки на основе метрик канала"
python3 "$ML_DIR/select_split.py"
SPLIT_LAYER=$(cat "$FILES_STORAGE/selected_split_layer.txt")

echo ">>> [2/6] Запуск edge.py (предобработка и разделение модели)"
python3 "$ML_DIR/edge.py" "$ML_DIR/mnist_images/7.png" --split_layer $SPLIT_LAYER

echo ">>> [3/6] Запуск симуляции передачи input_tensor.bin"
./ns3 run "scratch/wifi-sim --inputFile=$FILES_STORAGE/input_tensor_on_edge.bin --outputFile=$FILES_STORAGE/input_tensor_on_server.bin"

echo ">>> [4/6] Запуск server.py (обработка на серверной части)"
python3 "$ML_DIR/server.py"

echo ">>> [5/6] Запуск симуляции передачи result_tensor.bin обратно"
./ns3 run "scratch/wifi-sim --inputFile=$FILES_STORAGE/result_tensor_on_server.bin --outputFile=$FILES_STORAGE/result_tensor_on_edge.bin"

echo ">>> [6/6] Запуск edge_postprocess.py (финальное предсказание)"
python3 "$ML_DIR/edge_postprocess.py"
