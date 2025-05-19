set -e

ROOT_DIR="$(pwd)"
ML_DIR="$ROOT_DIR/ml"
FILES_STORAGE="$ROOT_DIR/files"
IMAGES_DIR="$ML_DIR/mnist_images"

generate_random_m() {
    echo "scale=2; $RANDOM % 200 / 100 + 0.5" | bc
}

rm -f "$FILES_STORAGE"/prediction_rand_results.csv
rm -f "$FILES_STORAGE"/energy_rand_log.csv
rm -f "$FILES_STORAGE"/latency_rand_log.csv

DISTANCES=(20 40 60 80 100)

for DIST in "${DISTANCES[@]}"; do
    for IMAGE_PATH in "$IMAGES_DIR"/*.png; do
    GT_LABEL=$(basename "$IMAGE_PATH" .png)

        M0=$(generate_random_m)
        M1=$(generate_random_m)
        M2=$(generate_random_m)

        rm -f "$FILES_STORAGE"/*.log "$FILES_STORAGE"/*.bin "$FILES_STORAGE"/selected_split_layer.txt 
        
        echo "=== Запуск эксперимента: image=$GT_LABEL, distance=${DIST}м ==="

        echo ">>> [0/6] Симуляция оценки канала"
        echo "TEST" > "$FILES_STORAGE/dummy_test.bin"
        ./ns3 run "scratch/wifi-sim --distance=$DIST --m0=$M0 --m1=$M1 --m2=$M2 --inputFile=$FILES_STORAGE/dummy_test.bin --outputFile=$FILES_STORAGE/dummy_result.bin"

        echo ">>> [1/6] Выбор split-точки"
        python3 "$ML_DIR/select_split_rand.py"
        SPLIT_LAYER=$(cat "$FILES_STORAGE/selected_split_layer.txt")

        rm -f "$FILES_STORAGE"/latency_transfer.log

        echo ">>> [2/6] Edge обработка"
        python3 "$ML_DIR/edge.py" "$IMAGE_PATH" --split_layer "$SPLIT_LAYER"

        echo ">>> [3/6] Симуляция передачи на сервер"
        ./ns3 run "scratch/wifi-sim --distance=$DIST --m0=$M0 --m1=$M1 --m2=$M2 --inputFile=$FILES_STORAGE/input_tensor_on_edge.bin --outputFile=$FILES_STORAGE/input_tensor_on_server.bin"

        echo ">>> [4/6] Cloud обработка"
        python3 "$ML_DIR/server.py"

        echo ">>> [5/6] Симуляция возврата результата"
        ./ns3 run "scratch/wifi-sim --distance=$DIST --m0=$M0 --m1=$M1 --m2=$M2 --inputFile=$FILES_STORAGE/result_tensor_on_server.bin --outputFile=$FILES_STORAGE/result_tensor_on_edge.bin"

        echo ">>> [6/6] Постобработка"
        python3 "$ML_DIR/edge_postprocess.py" "$FILES_STORAGE/latency_rand_log.csv" "$FILES_STORAGE/prediction_rand_results.csv"
    done
done

rm -f "$FILES_STORAGE"/*.log "$FILES_STORAGE"/*.bin "$FILES_STORAGE"/selected_split_layer.txt 