source ./venv/bin/activate
for batch_size in 16 64 128
do
  python main.py -n "$1" --batch-size $batch_size --use-cuda --force-reload
  echo  "Done $1 with batch-size $batch_size"
done