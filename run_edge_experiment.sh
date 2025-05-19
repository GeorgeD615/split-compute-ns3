set -e

ROOT_DIR="$(pwd)"
ML_DIR="$ROOT_DIR/ml"
FILES_STORAGE="$ROOT_DIR/files"
IMAGES_DIR="$ML_DIR/mnist_images"

DISTANCES=(20 40 60 80 100)

rm -f "$FILES_STORAGE"/prediction_results_full_edge.csv
rm -f "$FILES_STORAGE"/energy_full_edge_log.csv
rm -f "$FILES_STORAGE"/latency_full_edge_log.csv
for DIST in "${DISTANCES[@]}"; do
    for IMAGE_PATH in "$IMAGES_DIR"/*.png; do
        GT_LABEL=$(basename "$IMAGE_PATH" .png)
        echo "=== Запуск эксперимента: image=$GT_LABEL ==="
        echo ">>> [1/1] Edge обработка"
        python3 "$ML_DIR/edge_full_inference.py" "$IMAGE_PATH"
    done
done


rm -f "$FILES_STORAGE"/*.log "$FILES_STORAGE"/*.bin "$FILES_STORAGE"/selected_split_layer.txt